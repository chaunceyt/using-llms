// Package provisioner turns the configured defaults into concrete Kubernetes
// resources and creates them in a target namespace. All objects are built with
// the unstructured API so the set of resources (and their values) can change
// without touching the API types.
package provisioner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"

	"github.com/example/namespace-watcher/internal/config"
)

// Provisioner creates the default resources for a namespace.
type Provisioner struct {
	client dynamic.Interface
	mapper meta.RESTMapper
	cfg    config.Defaults
	log    *slog.Logger

	mu   sync.Mutex
	done map[string]struct{}
}

// New builds a Provisioner. The dynamic client performs the creates and the
// RESTMapper resolves each unstructured object's group/version/kind to the
// group/version/resource path needed by the dynamic client.
func New(client dynamic.Interface, mapper meta.RESTMapper, cfg config.Defaults, log *slog.Logger) *Provisioner {
	if log == nil {
		log = slog.Default()
	}
	return &Provisioner{
		client: client,
		mapper: mapper,
		cfg:    cfg,
		log:    log,
		done:   make(map[string]struct{}),
	}
}

// Provision creates every configured resource in the given namespace. It is
// idempotent: a namespace is provisioned at most once, and individual objects
// that already exist are skipped.
func (p *Provisioner) Provision(ctx context.Context, namespace string) error {
	p.mu.Lock()
	if _, ok := p.done[namespace]; ok {
		p.mu.Unlock()
		p.log.Debug("skipping already-provisioned namespace", "namespace", namespace)
		return nil
	}
	p.done[namespace] = struct{}{}
	p.mu.Unlock()

	p.log.Info("provisioning namespace", "namespace", namespace)

	var errs []error
	for _, obj := range Objects(namespace, p.cfg) {
		if err := p.createObject(ctx, obj); err != nil {
			errs = append(errs, fmt.Errorf("%s/%s: %w", obj.GetKind(), obj.GetName(), err))
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}

	p.log.Info("namespace provisioned",
		"namespace", namespace,
		"resources", len(Objects(namespace, p.cfg)),
	)
	return nil
}

// MarkDone records namespaces that should not be provisioned (used in
// new-only mode so pre-existing namespaces are ignored).
func (p *Provisioner) MarkDone(names ...string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, n := range names {
		p.done[n] = struct{}{}
	}
}

// IsDone reports whether a namespace has already been handled.
func (p *Provisioner) IsDone(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.done[name]
	return ok
}

// createObject resolves an unstructured object's GVR and creates it. An
// already-exists result is treated as success so provisioning stays idempotent
// even if MarkDone/IsDone bookkeeping is bypassed.
func (p *Provisioner) createObject(ctx context.Context, obj *unstructured.Unstructured) error {
	gvk := obj.GroupVersionKind()
	mapping, err := p.mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return fmt.Errorf("resolve rest mapping: %w", err)
	}

	_, err = p.client.Resource(mapping.Resource).Namespace(obj.GetNamespace()).
		Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			p.log.Debug("resource already exists, skipping",
				"kind", obj.GetKind(), "name", obj.GetName())
			return nil
		}
		return err
	}

	p.log.Info("created resource",
		"kind", obj.GetKind(), "name", obj.GetName(), "namespace", obj.GetNamespace())
	return nil
}
