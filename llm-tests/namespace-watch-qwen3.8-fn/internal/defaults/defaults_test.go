package defaults

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func testNS() *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "production", UID: "9a3d-01"},
	}
}

func find(t *testing.T, kind string) *unstructured.Unstructured {
	t.Helper()
	for _, r := range ForNamespace(testNS()) {
		if r.Body.GetKind() == kind {
			return r.Body
		}
	}
	t.Fatalf("no %s in defaults", kind)
	return nil
}

func TestLimitRangeValues(t *testing.T) {
	u := find(t, "LimitRange")
	limits, _, err := unstructured.NestedSlice(u.Object, "spec", "limits")
	if err != nil || len(limits) != 1 {
		t.Fatalf("spec.limits: %v, %v", limits, err)
	}
	first := limits[0].(map[string]interface{})
	if first["type"] != "Container" {
		t.Errorf("type: got %v", first["type"])
	}
	for _, key := range []string{"default", "defaultRequest", "max"} {
		block := first[key].(map[string]interface{})
		if _, ok := block["cpu"]; !ok {
			t.Errorf("%s: missing cpu", key)
		}
	}
	if got := first["default"].(map[string]interface{})["memory"]; got != "256Mi" {
		t.Errorf("default memory: got %v, want 256Mi", got)
	}
}

func TestResourceQuotaHardValues(t *testing.T) {
	u := find(t, "ResourceQuota")
	hard, _, err := unstructured.NestedStringMap(u.Object, "spec", "hard")
	if err != nil {
		t.Fatalf("spec.hard: %v", err)
	}
	want := map[string]string{
		"requests.cpu": "20", "requests.memory": "40Gi",
		"limits.cpu": "40", "limits.memory": "80Gi",
		"pods": "100", "services": "20", "persistentvolumeclaims": "30",
	}
	if len(hard) != len(want) {
		t.Fatalf("hard: got %d entries, want %d", len(hard), len(want))
	}
	for k, v := range want {
		if hard[k] != v {
			t.Errorf("hard[%s]: got %q, want %q", k, hard[k], v)
		}
	}
}

func TestRoleIsReadonlyConfigmaps(t *testing.T) {
	u := find(t, "Role")
	rules, _, err := unstructured.NestedSlice(u.Object, "rules")
	if err != nil || len(rules) != 1 {
		t.Fatalf("rules: %v, %v", rules, err)
	}
	rule := rules[0].(map[string]interface{})
	verbs := rule["verbs"].([]interface{})
	if len(verbs) != 2 || verbs[0] != "get" || verbs[1] != "list" {
		t.Errorf("verbs: got %v, want [get list]", verbs)
	}
}

func TestNetworkPolicyPorts(t *testing.T) {
	u := find(t, "NetworkPolicy")
	ingress, _, err := unstructured.NestedSlice(u.Object, "spec", "ingress")
	if err != nil || len(ingress) != 1 {
		t.Fatalf("ingress: %v, %v", ingress, err)
	}
	egress, _, err := unstructured.NestedSlice(u.Object, "spec", "egress")
	if err != nil || len(egress) != 2 {
		t.Fatalf("egress: %v, %v", egress, err)
	}
	dns := egress[1].(map[string]interface{})
	ports := dns["ports"].([]interface{})
	port := ports[0].(map[string]interface{})
	if port["port"] != int64(53) || port["protocol"] != "UDP" {
		t.Errorf("dns port: got %v", port)
	}
}

func TestAllResourcesOwnedAndNamespaced(t *testing.T) {
	ns := testNS()
	for _, r := range ForNamespace(ns) {
		if r.Body.GetNamespace() != ns.Name {
			t.Errorf("%s: namespace %q", r.Body.GetKind(), r.Body.GetNamespace())
		}
		refs := r.Body.GetOwnerReferences()
		if len(refs) != 1 || refs[0].Kind != "Namespace" || refs[0].Name != ns.Name {
			t.Errorf("%s: owner refs %v", r.Body.GetKind(), refs)
		}
		if r.Body.GetAPIVersion() == "" || r.Body.GetKind() == "" {
			t.Errorf("%s: missing apiVersion/kind", r.Body.GetName())
		}
	}
}
