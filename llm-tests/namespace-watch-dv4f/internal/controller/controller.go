// Package controller implements a Namespace watcher that provisions default
// resources into every newly created namespace.
package controller

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"namespace-watch/internal/resources"
)

// Controller watches for Namespace creation events and provisions the default
// resources via a Provisioner. A rate-limited work queue decouples event
// delivery from reconciliation, providing retry/backoff on transient failures.
type Controller struct {
	provisioner *resources.Provisioner

	namespaceLister corelisters.NamespaceLister
	namespaceSynced cache.InformerSynced
	workqueue       workqueue.TypedRateLimitingInterface[string]
}

// NewController wires the informer event handlers, lister and work queue.
func NewController(
	namespaceInformer coreinformers.NamespaceInformer,
	provisioner *resources.Provisioner,
) *Controller {
	c := &Controller{
		provisioner:     provisioner,
		namespaceLister: namespaceInformer.Lister(),
		namespaceSynced: namespaceInformer.Informer().HasSynced,
		workqueue:       workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[string](), workqueue.TypedRateLimitingQueueConfig[string]{Name: "namespaces"}),
	}

	namespaceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.handleAdd,
		UpdateFunc: c.handleUpdate,
	})

	return c
}

// Run starts the controller and blocks until ctx is cancelled.
func (c *Controller) Run(ctx context.Context, workers int) error {
	defer utilruntime.HandleCrash()
	defer c.workqueue.ShutDown()

	klog.Info("starting namespace controller")
	if !cache.WaitForCacheSync(ctx.Done(), c.namespaceSynced) {
		return fmt.Errorf("failed to sync namespace informer cache")
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}
	klog.Info("namespace controller started")

	<-ctx.Done()
	klog.Info("shutting down namespace controller")
	return nil
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	name, shutdown := c.workqueue.Get()
	if shutdown {
		return false
	}
	defer c.workqueue.Done(name)

	if err := c.syncNamespace(ctx, name); err != nil {
		klog.ErrorS(err, "failed to reconcile namespace", "namespace", name)
		// Re-queue with backoff; a transient API error must be retried.
		c.workqueue.AddRateLimited(name)
		return true
	}

	c.workqueue.Forget(name)
	return true
}

func (c *Controller) syncNamespace(ctx context.Context, name string) error {
	ns, err := c.namespaceLister.Get(name)
	if err != nil {
		// The namespace is gone; nothing to provision.
		return nil
	}
	// Skip system namespaces that we should never touch.
	if isExcluded(ns) {
		klog.V(2).Infof("skipping excluded namespace %q", name)
		return nil
	}

	if err := c.provisioner.Apply(ctx, ns.Name); err != nil {
		return fmt.Errorf("provisioning namespace %q: %w", ns.Name, err)
	}
	return nil
}

func (c *Controller) handleAdd(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err == nil {
		c.workqueue.Add(key)
	}
}

func (c *Controller) handleUpdate(oldObj, newObj interface{}) {
	// We provision only on creation; updates to existing namespaces are ignored
	// so the watcher does not fight with operators over resource changes.
	_ = oldObj
	if _, ok := newObj.(*corev1.Namespace); !ok {
		return
	}
}

// isExcluded reports whether a namespace must never be auto-provisioned.
func isExcluded(ns *corev1.Namespace) bool {
	switch ns.Name {
	case "default", "kube-system", "kube-public", "kube-node-lease":
		return true
	}
	return false
}
