// Package controller watches namespaces and ensures the default resources
// exist in each newly created namespace.
package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"namespace-watch-dv4f/internal/defaults"
)

// Controller reconciles namespace creation events into a set of default
// resources created through the dynamic client.
type Controller struct {
	dynamic      dynamic.Interface
	skipPrefixes []string

	informer cache.SharedIndexInformer
	synced   []cache.InformerSynced
	queue    workqueue.RateLimitingInterface
}

func New(
	dyn dynamic.Interface,
	informerFactory kubeinformers.SharedInformerFactory,
	skipPrefixes []string,
) *Controller {
	nsInformer := informerFactory.Core().V1().Namespaces().Informer()
	return &Controller{
		dynamic:      dyn,
		skipPrefixes: skipPrefixes,
		informer:     nsInformer,
		synced:       []cache.InformerSynced{nsInformer.HasSynced},
		queue:        workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
	}
}

func (c *Controller) Run(ctx context.Context, workers int) error {
	defer c.queue.ShutDown()

	_, err := c.informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			ns, ok := obj.(*corev1.Namespace)
			if !ok {
				return
			}
			c.enqueue(ns)
		},
	})
	if err != nil {
		return fmt.Errorf("registering namespace event handlers: %w", err)
	}

	klog.InfoS("starting namespace controller", "workers", workers)
	go c.informer.Run(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), c.synced...) {
		return fmt.Errorf("timed out waiting for informer caches to sync")
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.worker, time.Second)
	}

	<-ctx.Done()
	klog.InfoS("namespace controller stopped")
	return nil
}

func (c *Controller) enqueue(ns *corev1.Namespace) {
	if ns.Status.Phase != corev1.NamespaceActive {
		return
	}
	for _, prefix := range c.skipPrefixes {
		if strings.HasPrefix(ns.Name, prefix) {
			return
		}
	}
	klog.V(2).InfoS("enqueueing namespace", "namespace", ns.Name)
	c.queue.Add(ns.Name)
}

func (c *Controller) worker(ctx context.Context) {
	for c.processNext(ctx) {
	}
}

func (c *Controller) processNext(ctx context.Context) bool {
	key, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(key)

	err := c.sync(ctx, key.(string))
	switch {
	case err == nil:
		c.queue.Forget(key)
	case apierrors.IsNotFound(err):
		// Namespace deleted before we could act on it.
		c.queue.Forget(key)
	default:
		klog.ErrorS(err, "sync failed, requeueing", "namespace", key)
		c.queue.AddRateLimited(key)
	}
	return true
}

// sync is idempotent: missing defaults are created, existing ones are left
// untouched so operators can adjust the values per namespace.
func (c *Controller) sync(ctx context.Context, name string) error {
	obj, exists, err := c.informer.GetIndexer().GetByKey(name)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	ns := obj.(*corev1.Namespace)

	for _, res := range defaults.ForNamespace(ns) {
		_, err := c.dynamic.Resource(res.GVR).Namespace(ns.Name).
			Create(ctx, res.Body, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("creating %s/%s in namespace %s: %w",
				res.Body.GetKind(), res.Body.GetName(), ns.Name, err)
		}
		if err == nil {
			klog.InfoS("created default resource",
				"namespace", ns.Name, "kind", res.Body.GetKind(), "name", res.Body.GetName())
		}
	}
	return nil
}
