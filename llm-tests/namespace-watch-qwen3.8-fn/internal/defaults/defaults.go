// Package defaults builds the per-namespace default resources as unstructured
// objects. All values here are sane starting points intended to be adjusted
// per namespace after initialization.
package defaults

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	CoreV1 = func(resource string) schema.GroupVersionResource {
		return schema.GroupVersionResource{Group: "", Version: "v1", Resource: resource}
	}
	RBACV1 = func(resource string) schema.GroupVersionResource {
		return schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: resource}
	}
	NetworkingV1 = func(resource string) schema.GroupVersionResource {
		return schema.GroupVersionResource{Group: "networking.k8s.io", Version: "v1", Resource: resource}
	}
)

// Resource pairs a GroupVersionResource with the desired unstructured object
// so the caller can create it through the dynamic client.
type Resource struct {
	GVR  schema.GroupVersionResource
	Body *unstructured.Unstructured
}

// ForNamespace returns every default resource that should exist in the given
// namespace. The namespace object is used for metadata and owner references,
// so the created resources are garbage collected with the namespace.
func ForNamespace(ns *corev1.Namespace) []Resource {
	return []Resource{
		{CoreV1("limitranges"), limitRange(ns)},
		{CoreV1("resourcequotas"), resourceQuota(ns)},
		{CoreV1("serviceaccounts"), serviceAccount(ns)},
		{RBACV1("roles"), role(ns)},
		{RBACV1("rolebindings"), roleBinding(ns)},
		{NetworkingV1("networkpolicies"), networkPolicy(ns)},
	}
}

func limitRange(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "LimitRange",
		"metadata": map[string]interface{}{
			"name": "default-limits",
		},
		"spec": map[string]interface{}{
			"limits": []interface{}{
				map[string]interface{}{
					"type": "Container",
					"default": map[string]interface{}{
						"memory": "256Mi",
						"cpu":    "500m",
					},
					"defaultRequest": map[string]interface{}{
						"memory": "128Mi",
						"cpu":    "100m",
					},
					"max": map[string]interface{}{
						"memory": "2Gi",
						"cpu":    "2",
					},
				},
			},
		},
	}}
	ownedBy(u, ns)
	return u
}

func resourceQuota(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ResourceQuota",
		"metadata": map[string]interface{}{
			"name": "production-quota",
		},
		"spec": map[string]interface{}{
			"hard": map[string]interface{}{
				"requests.cpu":           "20",
				"requests.memory":        "40Gi",
				"limits.cpu":             "40",
				"limits.memory":          "80Gi",
				"pods":                   "100",
				"services":               "20",
				"persistentvolumeclaims": "30",
			},
		},
	}}
	ownedBy(u, ns)
	return u
}

func serviceAccount(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ServiceAccount",
		"metadata": map[string]interface{}{
			"name": "myapp-sa",
		},
	}}
	ownedBy(u, ns)
	return u
}

func role(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "Role",
		"metadata": map[string]interface{}{
			"name": "myapp-role",
		},
		"rules": []interface{}{
			map[string]interface{}{
				"apiGroups": []interface{}{""},
				"resources": []interface{}{"configmaps"},
				"verbs":     []interface{}{"get", "list"},
			},
		},
	}}
	ownedBy(u, ns)
	return u
}

func roleBinding(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "RoleBinding",
		"metadata": map[string]interface{}{
			"name": "myapp-rolebinding",
		},
		"subjects": []interface{}{
			map[string]interface{}{
				"kind":      "ServiceAccount",
				"name":      "myapp-sa",
				"namespace": ns.Name,
			},
		},
		"roleRef": map[string]interface{}{
			"kind":     "Role",
			"apiGroup": "rbac.authorization.k8s.io",
			"name":     "myapp-role",
		},
	}}
	ownedBy(u, ns)
	return u
}

func networkPolicy(ns *corev1.Namespace) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "networking.k8s.io/v1",
		"kind":       "NetworkPolicy",
		"metadata": map[string]interface{}{
			"name": "myapp-network-policy",
		},
		"spec": map[string]interface{}{
			"podSelector": map[string]interface{}{
				"matchLabels": map[string]interface{}{
					"app": "myapp",
				},
			},
			"policyTypes": []interface{}{"Ingress", "Egress"},
			"ingress": []interface{}{
				map[string]interface{}{
					"from": []interface{}{
						map[string]interface{}{
							"podSelector": map[string]interface{}{
								"matchLabels": map[string]interface{}{
									"app": "nginx-ingress",
								},
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
								"matchLabels": map[string]interface{}{
									"app": "postgres",
								},
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
	ownedBy(u, ns)
	return u
}

func ownedBy(u *unstructured.Unstructured, owner *corev1.Namespace) {
	u.SetNamespace(owner.Name)
	u.SetOwnerReferences([]metav1.OwnerReference{
		*metav1.NewControllerRef(owner, corev1.SchemeGroupVersion.WithKind("Namespace")),
	})
}
