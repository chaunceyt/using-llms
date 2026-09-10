package resources

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// NetworkPolicyGVR identifies the networking.k8s.io/v1 NetworkPolicy resource.
var NetworkPolicyGVR = schema.GroupVersionResource{
	Group:    "networking.k8s.io",
	Version:  "v1",
	Resource: "networkpolicies",
}

const NetworkPolicyName = "myapp-network-policy"

// BuildNetworkPolicy constructs a default deny-by-default policy: ingress only
// from the nginx-ingress pods on port 3000, egress only to postgres (5432) and
// cluster DNS (UDP 53).
func BuildNetworkPolicy(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "networking.k8s.io/v1",
		"kind":       "NetworkPolicy",
		"metadata": map[string]interface{}{
			"name":      NetworkPolicyName,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"podSelector": map[string]interface{}{
				"matchLabels": map[string]interface{}{"app": "myapp"},
			},
			"policyTypes": []interface{}{"Ingress", "Egress"},
			"ingress": []interface{}{
				map[string]interface{}{
					"from": []interface{}{
						map[string]interface{}{
							"podSelector": map[string]interface{}{
								"matchLabels": map[string]interface{}{"app": "nginx-ingress"},
							},
						},
					},
					"ports": []interface{}{
						map[string]interface{}{
							"protocol": "TCP",
							"port":     int64(3000),
						},
					},
				},
			},
			"egress": []interface{}{
				map[string]interface{}{
					"to": []interface{}{
						map[string]interface{}{
							"podSelector": map[string]interface{}{
								"matchLabels": map[string]interface{}{"app": "postgres"},
							},
						},
					},
					"ports": []interface{}{
						map[string]interface{}{
							"protocol": "TCP",
							"port":     int64(5432),
						},
					},
				},
				map[string]interface{}{
					"to": []interface{}{
						map[string]interface{}{
							"namespaceSelector": map[string]interface{}{},
						},
					},
					"ports": []interface{}{
						map[string]interface{}{
							"protocol": "UDP",
							"port":     int64(53),
						},
					},
				},
			},
		},
	}}
}
