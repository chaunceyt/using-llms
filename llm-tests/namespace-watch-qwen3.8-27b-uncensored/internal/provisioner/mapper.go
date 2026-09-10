package provisioner

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/example/namespace-watcher/internal/config"
)

// GVKs returns the distinct group/version/kinds of the resources the
// provisioner creates, derived from the configured object set. The set is
// fixed by design (only the values change), which is what makes a static REST
// mapper safe to use in place of a runtime discovery call.
func GVKs(cfg config.Defaults) []schema.GroupVersionKind {
	seen := make(map[schema.GroupVersionKind]struct{})
	var out []schema.GroupVersionKind
	for _, obj := range Objects("placeholder", cfg) {
		gvk := obj.GroupVersionKind()
		if _, ok := seen[gvk]; ok {
			continue
		}
		seen[gvk] = struct{}{}
		out = append(out, gvk)
	}
	return out
}

// NewStaticMapper returns a meta.RESTMapper that knows how to resolve the
// provisioner's fixed set of resources to their REST resources. It avoids a
// startup discovery call and the associated RBAC requirements, while remaining
// fully correct for the resources the watcher creates.
func NewStaticMapper(cfg config.Defaults) meta.RESTMapper {
	mapper := meta.NewDefaultRESTMapper(nil)
	for _, gvk := range GVKs(cfg) {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}
	return mapper
}
