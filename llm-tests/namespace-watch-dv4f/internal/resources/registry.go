// Package resources defines the set of default objects provisioned whenever a
// new Kubernetes Namespace is created by the watcher. All objects are built as
// k8s.io/apimachinery/pkg/apis/meta/v1/unstructured.Unstructured so that no
// typed scheme is required, and applied via the dynamic client.
package resources

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Resource couples a GroupVersionResource (needed by the dynamic client) with
// a builder that produces the desired unstructured object for a given namespace.
type Resource struct {
	GVR   schema.GroupVersionResource
	Build func(namespace string) *unstructured.Unstructured
}

// Registry is an ordered collection of resources to provision on namespace
// creation. Order matters only for readability; each object is independent.
type Registry struct {
	Resources []Resource
}

// NewRegistry returns a Registry preloaded with every default resource.
func NewRegistry() *Registry {
	return &Registry{
		Resources: []Resource{
			{GVR: LimitRangeGVR, Build: BuildLimitRange},
			{GVR: ResourceQuotaGVR, Build: BuildResourceQuota},
			{GVR: ServiceAccountGVR, Build: BuildServiceAccount},
			{GVR: RoleGVR, Build: BuildRole},
			{GVR: RoleBindingGVR, Build: BuildRoleBinding},
			{GVR: NetworkPolicyGVR, Build: BuildNetworkPolicy},
		},
	}
}
