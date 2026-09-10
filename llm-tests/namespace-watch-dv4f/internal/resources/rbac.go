package resources

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupVersionResources for the RBAC-related resources.
var (
	ServiceAccountGVR = schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "serviceaccounts",
	}
	RoleGVR = schema.GroupVersionResource{
		Group:    "rbac.authorization.k8s.io",
		Version:  "v1",
		Resource: "roles",
	}
	RoleBindingGVR = schema.GroupVersionResource{
		Group:    "rbac.authorization.k8s.io",
		Version:  "v1",
		Resource: "rolebindings",
	}
)

// Identity names shared by the ServiceAccount, Role and RoleBinding.
const (
	ServiceAccountName = "myapp-sa"
	RoleName           = "myapp-role"
	RoleBindingName    = "myapp-rolebinding"
)

// BuildServiceAccount constructs the default application ServiceAccount.
func BuildServiceAccount(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ServiceAccount",
		"metadata": map[string]interface{}{
			"name":      ServiceAccountName,
			"namespace": namespace,
		},
	}}
}

// BuildRole constructs a minimal read-only Role scoped to ConfigMaps.
func BuildRole(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "Role",
		"metadata": map[string]interface{}{
			"name":      RoleName,
			"namespace": namespace,
		},
		"rules": []interface{}{
			map[string]interface{}{
				"apiGroups": []interface{}{""},
				"resources": []interface{}{"configmaps"},
				"verbs":     []interface{}{"get", "list"},
			},
		},
	}}
}

// BuildRoleBinding binds the ServiceAccount to the Role.
func BuildRoleBinding(namespace string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "rbac.authorization.k8s.io/v1",
		"kind":       "RoleBinding",
		"metadata": map[string]interface{}{
			"name":      RoleBindingName,
			"namespace": namespace,
		},
		"subjects": []interface{}{
			map[string]interface{}{
				"kind":      "ServiceAccount",
				"name":      ServiceAccountName,
				"namespace": namespace,
			},
		},
		"roleRef": map[string]interface{}{
			"kind":     "Role",
			"apiGroup": "rbac.authorization.k8s.io",
			"name":     RoleName,
		},
	}}
}
