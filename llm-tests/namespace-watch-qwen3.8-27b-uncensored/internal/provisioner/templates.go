package provisioner

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/example/namespace-watcher/internal/config"
)

// stringMap converts a map[string]string into a map[string]interface{} so it
// can be embedded directly inside an unstructured object.
func stringMap(m map[string]string) map[string]interface{} {
	out := make(map[string]interface{}, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// stringSlice converts a []string into a []interface{} so it can be embedded
// inside an unstructured object. The unstructured package requires JSON value
// types ([]interface{}, not []string).
func stringSlice(s []string) []interface{} {
	out := make([]interface{}, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}

// metadata builds the standard metadata block for a namespaced resource,
// stamping the configured labels onto it.
func metadata(name, namespace string, labels map[string]string) map[string]interface{} {
	md := map[string]interface{}{
		"name":      name,
		"namespace": namespace,
	}
	if len(labels) > 0 {
		md["labels"] = stringMap(labels)
	}
	return md
}

// NewLimitRange returns the unstructured LimitRange for a namespace.
func NewLimitRange(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	lr := cfg.LimitRange
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "LimitRange",
		"metadata":   metadata(lr.Name, namespace, cfg.Labels),
		"spec": map[string]interface{}{
			"limits": []interface{}{
				map[string]interface{}{
					"type": "Container",
					"default": map[string]interface{}{
						"memory": lr.DefaultMemory,
						"cpu":    lr.DefaultCPU,
					},
					"defaultRequest": map[string]interface{}{
						"memory": lr.DefaultRequestMemory,
						"cpu":    lr.DefaultRequestCPU,
					},
					"max": map[string]interface{}{
						"memory": lr.MaxMemory,
						"cpu":    lr.MaxCPU,
					},
				},
			},
		},
	}}
}

// NewResourceQuota returns the unstructured ResourceQuota for a namespace.
func NewResourceQuota(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	q := cfg.ResourceQuota
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ResourceQuota",
		"metadata":   metadata(q.Name, namespace, cfg.Labels),
		"spec": map[string]interface{}{
			"hard": map[string]interface{}{
				"requests.cpu":           q.RequestsCPU,
				"requests.memory":        q.RequestsMemory,
				"limits.cpu":             q.LimitsCPU,
				"limits.memory":          q.LimitsMemory,
				"pods":                   q.Pods,
				"services":               q.Services,
				"persistentvolumeclaims": q.PersistentVolumeClaims,
			},
		},
	}}
}

// NewServiceAccount returns the unstructured ServiceAccount for a namespace.
func NewServiceAccount(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ServiceAccount",
		"metadata":   metadata(cfg.RBAC.ServiceAccountName, namespace, cfg.Labels),
	}}
}

// NewRole returns the unstructured Role granting read-only access to the
// configured resources.
func NewRole(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	r := cfg.RBAC
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "Role",
		"metadata":   metadata(r.RoleName, namespace, cfg.Labels),
		"rules": []interface{}{
			map[string]interface{}{
				"apiGroups": stringSlice(r.APIGroups),
				"resources": stringSlice(r.Resources),
				"verbs":     stringSlice(r.Verbs),
			},
		},
	}}
}

// NewRoleBinding binds the Role to the ServiceAccount within the namespace.
func NewRoleBinding(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	r := cfg.RBAC
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "RoleBinding",
		"metadata":   metadata(r.RoleBindingName, namespace, cfg.Labels),
		"subjects": []interface{}{
			map[string]interface{}{
				"kind":      "ServiceAccount",
				"name":      r.ServiceAccountName,
				"namespace": namespace,
			},
		},
		"roleRef": map[string]interface{}{
			"kind":     "Role",
			"apiGroup": "rbac.authorization.k8s.io",
			"name":     r.RoleName,
		},
	}}
}

// NewNetworkPolicy returns the unstructured NetworkPolicy for a namespace.
func NewNetworkPolicy(namespace string, cfg config.Defaults) *unstructured.Unstructured {
	np := cfg.NetworkPolicy

	egress := []interface{}{
		map[string]interface{}{
			"to": []interface{}{
				map[string]interface{}{
					"podSelector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"app": np.Egress.ToApp,
						},
					},
				},
			},
			"ports": []interface{}{
				map[string]interface{}{
					"protocol": np.Egress.Protocol,
					"port":     int64(np.Egress.Port),
				},
			},
		},
	}
	if np.AllowDNS {
		egress = append(egress, map[string]interface{}{
			"to": []interface{}{
				map[string]interface{}{
					"namespaceSelector": map[string]interface{}{},
				},
			},
			"ports": []interface{}{
				map[string]interface{}{
					"protocol": "UDP",
					"port":     int64(np.DNSPort),
				},
			},
		})
	}

	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "networking.k8s.io/v1",
		"kind":       "NetworkPolicy",
		"metadata":   metadata(np.Name, namespace, cfg.Labels),
		"spec": map[string]interface{}{
			"podSelector": map[string]interface{}{
				"matchLabels": stringMap(np.PodSelector),
			},
			"policyTypes": stringSlice(np.PolicyTypes),
			"ingress": []interface{}{
				map[string]interface{}{
					"from": []interface{}{
						map[string]interface{}{
							"podSelector": map[string]interface{}{
								"matchLabels": map[string]interface{}{
									"app": np.Ingress.FromApp,
								},
							},
						},
					},
					"ports": []interface{}{
						map[string]interface{}{
							"protocol": np.Ingress.Protocol,
							"port":     int64(np.Ingress.Port),
						},
					},
				},
			},
			"egress": egress,
		},
	}}
}

// Objects returns the full ordered set of resources that should be created for
// a namespace, driven entirely by the provided defaults.
func Objects(namespace string, cfg config.Defaults) []*unstructured.Unstructured {
	return []*unstructured.Unstructured{
		NewLimitRange(namespace, cfg),
		NewResourceQuota(namespace, cfg),
		NewServiceAccount(namespace, cfg),
		NewRole(namespace, cfg),
		NewRoleBinding(namespace, cfg),
		NewNetworkPolicy(namespace, cfg),
	}
}
