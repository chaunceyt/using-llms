// Package watcher watches Kubernetes namespaces and triggers provisioning of
// the default resources whenever a namespace is created.
//
// It is built on a client-go SharedInformer. Namespace create events are fanned
// out to a bounded work queue and processed by a small pool of workers, so a
// burst of namespace creations never blocks the informer and provisioning is
// deduplicated per namespace.
package watcher

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"

	"github.com/example/namespace-watcher/internal/provisioner"
)

const defaultQueueSize = 1024

// Options configures the Watcher.
type Options struct {
	// Backfill, when true, also provisions namespaces that already exist when
	// the watcher starts. When false (the default) only namespaces created
	// after the watcher starts are provisioned.
	Backfill bool
	// Concurrency is the number of workers processing the work queue. Values
	// <= 0 default to 1.
	Concurrency int
	// Log is the logger. Defaults to slog.Default() when nil.
	Log *slog.Logger
}

// Watcher watches namespaces and provisions the default resources.
type Watcher struct {
	clientset   kubernetes.Interface
	factory     informers.SharedInformerFactory
	nsInformer  cache.SharedIndexInformer
	provisioner *provisioner.Provisioner
	log         *slog.Logger
	backfill    bool
	concurrency int

	// known holds namespaces that existed before the watcher started. It is
	// only populated in new-only mode and used to exclude pre-existing ones.
	known map[string]struct{}

	queue    chan string
	inflight map[string]struct{}
	queueMu  sync.Mutex

	// ctx is the watcher's lifetime context, set in Run. Event handlers can
	// only fire after Run starts the informer, so it is always initialized
	// before it is read.
	ctx context.Context
}

// New builds a Watcher on top of the given clientset and provisioner.
func New(clientset kubernetes.Interface, prov *provisioner.Provisioner, opts Options) *Watcher {
	if opts.Log == nil {
		opts.Log = slog.Default()
	}
	concurrency := opts.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	factory := informers.NewSharedInformerFactory(clientset, 0)
	return &Watcher{
		clientset:   clientset,
		factory:     factory,
		nsInformer:  factory.Core().V1().Namespaces().Informer(),
		provisioner: prov,
		log:         opts.Log,
		backfill:    opts.Backfill,
		concurrency: concurrency,
		known:       make(map[string]struct{}),
		inflight:    make(map[string]struct{}),
		queue:       make(chan string, defaultQueueSize),
	}
}

// Run starts the informer, waits for it to sync, starts the worker pool, and
// blocks until the context is cancelled.
func (w *Watcher) Run(ctx context.Context) error {
	w.ctx = ctx

	// Capture pre-existing namespaces before the informer starts so that, in
	// new-only mode, they are excluded from provisioning.
	if err := w.captureExisting(ctx); err != nil {
		return err
	}

	if _, err := w.nsInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: w.onAdd,
	}); err != nil {
		return fmt.Errorf("add namespace event handler: %w", err)
	}

	w.factory.Start(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), w.nsInformer.HasSynced) {
		return errors.New("failed to sync namespace informer cache")
	}
	w.log.Info("namespace informer synced",
		"mode", mode(w.backfill), "workers", w.concurrency)

	var wg sync.WaitGroup
	for i := 0; i < w.concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.worker(ctx)
		}()
	}
	// Block until the context is cancelled, then wait for the worker pool to
	// finish its in-flight work so the watcher shuts down cleanly.
	<-ctx.Done()
	wg.Wait()
	w.log.Info("watcher shutting down")
	return nil
}

// mode returns a human-readable name for the watcher mode.
func mode(backfill bool) string {
	if backfill {
		return "backfill"
	}
	return "new-only"
}

// captureExisting records the namespaces that already exist so new-only mode
// can ignore them. It is a no-op in backfill mode.
func (w *Watcher) captureExisting(ctx context.Context) error {
	if w.backfill {
		return nil
	}
	list, err := w.clientset.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list existing namespaces: %w", err)
	}
	for _, ns := range list.Items {
		w.known[ns.Name] = struct{}{}
	}
	w.log.Info("captured pre-existing namespaces", "count", len(list.Items))
	return nil
}

// onAdd is the informer event handler. It enqueues namespaces that need
// provisioning, applying the mode and deduplication rules.
func (w *Watcher) onAdd(obj interface{}) {
	ns, ok := obj.(*corev1.Namespace)
	if !ok {
		w.log.Warn("unexpected object type from namespace informer", "type", fmt.Sprintf("%T", obj))
		return
	}
	// Ignore namespaces that are already being deleted.
	if ns.DeletionTimestamp != nil {
		return
	}
	name := ns.Name

	// In new-only mode skip namespaces that existed before we started.
	if !w.backfill && w.isKnown(name) {
		return
	}

	// Claim the namespace for provisioning (dedup). The lock is held only long
	// enough to update inflight so the (possibly blocking) enqueue below does
	// not hold it.
	w.queueMu.Lock()
	if w.provisioner.IsDone(name) {
		w.queueMu.Unlock()
		return
	}
	if _, inflight := w.inflight[name]; inflight {
		w.queueMu.Unlock()
		return
	}
	w.inflight[name] = struct{}{}
	w.queueMu.Unlock()

	// Enqueue for a worker; blocks only if the queue is full, and drops only
	// on shutdown.
	select {
	case w.queue <- name:
	case <-w.ctxDone():
	}
}

// ctxDone returns a channel that is closed when the watcher stops, allowing
// event handlers to avoid blocking on enqueue after shutdown.
func (w *Watcher) ctxDone() <-chan struct{} {
	return w.ctx.Done()
}

// isKnown reports whether a namespace was captured at startup.
func (w *Watcher) isKnown(name string) bool {
	_, ok := w.known[name]
	return ok
}

// worker pulls namespace names from the queue and provisions them.
func (w *Watcher) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case name := <-w.queue:
			w.run(ctx, name)
		}
	}
}

// run provisions a single namespace and always clears its inflight marker.
func (w *Watcher) run(ctx context.Context, name string) {
	defer func() {
		w.queueMu.Lock()
		delete(w.inflight, name)
		w.queueMu.Unlock()
	}()
	if err := w.provisioner.Provision(ctx, name); err != nil {
		w.log.Error("failed to provision namespace", "namespace", name, "err", err)
	}
}
