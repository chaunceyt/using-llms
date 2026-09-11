package resources

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// LimitRangeGVR identifies the core v1 LimitRange resource.
var LimitRangeGVR = schema.GroupVersionResource{
	Group:    "",
	Version:  "v1",
	Resource: "limitranges",
}

// Default values applied to Containers that omit resource limits. Change these
// constants to alter what every new namespace receives.
const (
	LimitRangeName          = "default-limits"
	LimitRangeDefaultMemory = "256Mi"
	LimitRangeDefaultCPU    = "500m"
	LimitRangeRequestMemory = "128Mi"
	LimitRangeRequestCPU    = "100m"
	LimitRangeMaxMemory     = "2Gi"
	LimitRangeMaxCPU        = "2"
)

// BuildLimitRange constructs the default LimitRange for a namespace.
func BuildLimitRange(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "LimitRange",
		"metadata": map[string]interface{}{
			"name":      LimitRangeName,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"limits": []interface{}{
				map[string]interface{}{
					"type": "Container",
					"default": map[string]interface{}{
						"memory": LimitRangeDefaultMemory,
						"cpu":    LimitRangeDefaultCPU,
					},
					"defaultRequest": map[string]interface{}{
						"memory": LimitRangeRequestMemory,
						"cpu":    LimitRangeRequestCPU,
					},
					"max": map[string]interface{}{
						"memory": LimitRangeMaxMemory,
						"cpu":    LimitRangeMaxCPU,
					},
				},
			},
		},
	}}
}
