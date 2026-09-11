package resources

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ResourceQuotaGVR identifies the core v1 ResourceQuota resource.
var ResourceQuotaGVR = schema.GroupVersionResource{
	Group:    "",
	Version:  "v1",
	Resource: "resourcequotas",
}

// Default hard quota values applied to each new namespace.
const (
	ResourceQuotaName        = "production-quota"
	QuotaRequestsCPU         = "20"
	QuotaRequestsMemory      = "40Gi"
	QuotaLimitsCPU           = "40"
	QuotaLimitsMemory        = "80Gi"
	QuotaPods                = "100"
	QuotaServices            = "20"
	QuotaPersistentVolClaims = "30"
)

// BuildResourceQuota constructs the default ResourceQuota for a namespace.
func BuildResourceQuota(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ResourceQuota",
		"metadata": map[string]interface{}{
			"name":      ResourceQuotaName,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"hard": map[string]interface{}{
				"requests.cpu":           QuotaRequestsCPU,
				"requests.memory":        QuotaRequestsMemory,
				"limits.cpu":             QuotaLimitsCPU,
				"limits.memory":          QuotaLimitsMemory,
				"pods":                   QuotaPods,
				"services":               QuotaServices,
				"persistentvolumeclaims": QuotaPersistentVolClaims,
			},
		},
	}}
}
