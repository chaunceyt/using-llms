package provisioner

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/example/namespace-watcher/internal/config"
)

const testNS = "production"

// to converts an unstructured object into the typed API object T so the test
// can assert on well-known fields. A conversion failure means the unstructured
// object is malformed, which is exactly what we want to catch.
func to[T any](t *testing.T, obj *unstructured.Unstructured) T {
	t.Helper()
	var out T
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &out); err != nil {
		t.Fatalf("convert %s to %T: %v", obj.GetKind(), out, err)
	}
	return out
}

func assertBase(t *testing.T, obj *unstructured.Unstructured, wantGV, wantKind, wantName, wantNS string) {
	t.Helper()
	gvk := obj.GroupVersionKind()
	if got := gvk.GroupVersion().String(); got != wantGV {
		t.Errorf("apiVersion = %q, want %q", got, wantGV)
	}
	if gvk.Kind != wantKind {
		t.Errorf("kind = %q, want %q", gvk.Kind, wantKind)
	}
	if obj.GetName() != wantName {
		t.Errorf("name = %q, want %q", obj.GetName(), wantName)
	}
	if obj.GetNamespace() != wantNS {
		t.Errorf("namespace = %q, want %q", obj.GetNamespace(), wantNS)
	}
	if _, ok := obj.GetLabels()[config.DefaultLabelKey]; !ok {
		t.Errorf("expected managed-by label %q on %s", config.DefaultLabelKey, obj.GetName())
	}
}

// proto dereferences a network policy port protocol, tolerating a nil value.
func proto(p *corev1.Protocol) corev1.Protocol {
	if p == nil {
		return ""
	}
	return *p
}

func TestNewLimitRange(t *testing.T) {
	obj := NewLimitRange(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "v1", "LimitRange", "default-limits", testNS)

	lr := to[corev1.LimitRange](t, obj)
	if len(lr.Spec.Limits) != 1 {
		t.Fatalf("expected 1 limit entry, got %d", len(lr.Spec.Limits))
	}
	l := lr.Spec.Limits[0]
	if l.Type != corev1.LimitTypeContainer {
		t.Errorf("type = %q, want Container", l.Type)
	}
	q := func(rl corev1.ResourceList, name corev1.ResourceName) string {
		v, ok := rl[name]
		if !ok {
			return "<absent>"
		}
		return v.String()
	}
	check := func(list corev1.ResourceList, name corev1.ResourceName, want string) {
		t.Helper()
		if got := q(list, name); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
	check(l.Default, corev1.ResourceMemory, "256Mi")
	check(l.Default, corev1.ResourceCPU, "500m")
	check(l.DefaultRequest, corev1.ResourceMemory, "128Mi")
	check(l.DefaultRequest, corev1.ResourceCPU, "100m")
	check(l.Max, corev1.ResourceMemory, "2Gi")
	check(l.Max, corev1.ResourceCPU, "2")
}

func TestNewResourceQuota(t *testing.T) {
	obj := NewResourceQuota(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "v1", "ResourceQuota", "production-quota", testNS)

	rq := to[corev1.ResourceQuota](t, obj)
	hard := rq.Spec.Hard
	check := func(name string, want string) {
		t.Helper()
		got, ok := hard[corev1.ResourceName(name)]
		if !ok || got.String() != want {
			t.Errorf("hard[%s] = %q, want %q", name, got.String(), want)
		}
	}
	check("requests.cpu", "20")
	check("requests.memory", "40Gi")
	check("limits.cpu", "40")
	check("limits.memory", "80Gi")
	check("pods", "100")
	check("services", "20")
	check("persistentvolumeclaims", "30")
}

func TestNewServiceAccount(t *testing.T) {
	obj := NewServiceAccount(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "v1", "ServiceAccount", "myapp-sa", testNS)
	to[corev1.ServiceAccount](t, obj) // must convert cleanly
}

func TestNewRole(t *testing.T) {
	obj := NewRole(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "rbac.authorization.k8s.io/v1", "Role", "myapp-role", testNS)

	role := to[rbacv1.Role](t, obj)
	if len(role.Rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(role.Rules))
	}
	rule := role.Rules[0]
	if len(rule.APIGroups) != 1 || rule.APIGroups[0] != "" {
		t.Errorf("apiGroups = %v, want [\"\"]", rule.APIGroups)
	}
	if len(rule.Resources) != 1 || rule.Resources[0] != "configmaps" {
		t.Errorf("resources = %v, want [configmaps]", rule.Resources)
	}
	if len(rule.Verbs) != 2 || rule.Verbs[0] != "get" || rule.Verbs[1] != "list" {
		t.Errorf("verbs = %v, want [get list]", rule.Verbs)
	}
}

func TestNewRoleBinding(t *testing.T) {
	obj := NewRoleBinding(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "rbac.authorization.k8s.io/v1", "RoleBinding", "myapp-rolebinding", testNS)

	rb := to[rbacv1.RoleBinding](t, obj)
	if rb.RoleRef.Kind != "Role" || rb.RoleRef.APIGroup != "rbac.authorization.k8s.io" || rb.RoleRef.Name != "myapp-role" {
		t.Errorf("roleRef = %+v, want Role/rbac.authorization.k8s.io/myapp-role", rb.RoleRef)
	}
	if len(rb.Subjects) != 1 {
		t.Fatalf("expected 1 subject, got %d", len(rb.Subjects))
	}
	s := rb.Subjects[0]
	if s.Kind != "ServiceAccount" || s.Name != "myapp-sa" || s.Namespace != testNS {
		t.Errorf("subject = %+v, want ServiceAccount/myapp-sa in %s", s, testNS)
	}
}

func TestNewNetworkPolicy(t *testing.T) {
	obj := NewNetworkPolicy(testNS, config.NewDefaultConfig())
	assertBase(t, obj, "networking.k8s.io/v1", "NetworkPolicy", "myapp-network-policy", testNS)

	np := to[networkingv1.NetworkPolicy](t, obj)
	if np.Spec.PodSelector.MatchLabels["app"] != "myapp" {
		t.Errorf("podSelector = %v, want app=myapp", np.Spec.PodSelector.MatchLabels)
	}
	if len(np.Spec.PolicyTypes) != 2 {
		t.Errorf("policyTypes = %v, want 2 entries", np.Spec.PolicyTypes)
	}

	// Ingress: from app=nginx-ingress on TCP/3000.
	if len(np.Spec.Ingress) != 1 {
		t.Fatalf("expected 1 ingress rule, got %d", len(np.Spec.Ingress))
	}
	ing := np.Spec.Ingress[0]
	if len(ing.From) != 1 || ing.From[0].PodSelector.MatchLabels["app"] != "nginx-ingress" {
		t.Errorf("ingress.from = %+v, want podSelector app=nginx-ingress", ing.From)
	}
	if len(ing.Ports) != 1 || ing.Ports[0].Port.IntValue() != 3000 || proto(ing.Ports[0].Protocol) != corev1.ProtocolTCP {
		t.Errorf("ingress.ports = %+v, want TCP/3000", ing.Ports)
	}

	// Egress: to app=postgres on TCP/5432, plus DNS to any namespace on UDP/53.
	if len(np.Spec.Egress) != 2 {
		t.Fatalf("expected 2 egress rules (db + dns), got %d", len(np.Spec.Egress))
	}
	db := np.Spec.Egress[0]
	if len(db.To) != 1 || db.To[0].PodSelector.MatchLabels["app"] != "postgres" {
		t.Errorf("egress[0].to = %+v, want podSelector app=postgres", db.To)
	}
	if len(db.Ports) != 1 || db.Ports[0].Port.IntValue() != 5432 || proto(db.Ports[0].Protocol) != corev1.ProtocolTCP {
		t.Errorf("egress[0].ports = %+v, want TCP/5432", db.Ports)
	}

	dns := np.Spec.Egress[1]
	if len(dns.To) != 1 || dns.To[0].NamespaceSelector == nil {
		t.Errorf("egress[1].to = %+v, want namespaceSelector {}", dns.To)
	}
	if len(dns.Ports) != 1 || dns.Ports[0].Port.IntValue() != 53 || proto(dns.Ports[0].Protocol) != corev1.ProtocolUDP {
		t.Errorf("egress[1].ports = %+v, want UDP/53", dns.Ports)
	}
}

func TestObjects(t *testing.T) {
	objs := Objects(testNS, config.NewDefaultConfig())
	want := []string{"LimitRange", "ResourceQuota", "ServiceAccount", "Role", "RoleBinding", "NetworkPolicy"}
	if len(objs) != len(want) {
		t.Fatalf("expected %d objects, got %d", len(want), len(objs))
	}
	for i, o := range objs {
		if o.GetKind() != want[i] {
			t.Errorf("object[%d].kind = %q, want %q", i, o.GetKind(), want[i])
		}
		if o.GetNamespace() != testNS {
			t.Errorf("object[%d].namespace = %q, want %q", i, o.GetNamespace(), testNS)
		}
	}
}
