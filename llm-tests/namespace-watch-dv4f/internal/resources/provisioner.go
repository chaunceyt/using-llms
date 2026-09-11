package resources

import (
	"context"
	"errors"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog/v2"
)

// Provisioner applies the Registry's resources to namespaces using the dynamic
// client. Objects are created only when absent so that operators can safely
// modify or replace them afterwards without the watcher clobbering their work.
type Provisioner struct {
	dynamicClient dynamic.Interface
	registry      *Registry
}

// NewProvisioner returns a Provisioner backed by the given dynamic client.
func NewProvisioner(client dynamic.Interface, registry *Registry) *Provisioner {
	return &Provisioner{
		dynamicClient: client,
		registry:      registry,
	}
}

// Apply creates every registered resource inside namespace unless it already
// exists. It aggregates per-resource failures and returns them as one error.
func (p *Provisioner) Apply(ctx context.Context, namespace string) error {
	var errs []error
	for _, res := range p.registry.Resources {
		if err := p.applyOne(ctx, namespace, res); err != nil {
			errs = append(errs, fmt.Errorf("%s %q: %w", res.GVR.Resource, namespace, err))
		}
	}
	return errors.Join(errs...)
}

// applyOne creates a single resource, tolerating an already-existing object.
func (p *Provisioner) applyOne(ctx context.Context, namespace string, res Resource) error {
	obj := res.Build(namespace)
	name := obj.GetName()

	if _, err := p.dynamicClient.Resource(res.GVR).Namespace(namespace).
		Create(ctx, obj, metav1.CreateOptions{}); err != nil {
		if apierrors.IsAlreadyExists(err) {
			klog.V(2).Infof("skipping %s %q/%q: already exists", res.GVR.Resource, namespace, name)
			return nil
		}
		return err
	}

	klog.Infof("created %s %q/%q in namespace %q", obj.GetKind(), name, namespace, res.GVR.GroupVersion())
	return nil
}
