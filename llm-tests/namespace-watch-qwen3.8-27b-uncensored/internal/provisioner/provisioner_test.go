package provisioner

import (
	"context"
	"io"
	"log/slog"
	"testing"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"

	"github.com/example/namespace-watcher/internal/config"
)

// discardLogger is a logger that swallows output for quiet tests.
var discardLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// newFakeProvisioner wires a Provisioner against a fake dynamic client and a
// static RESTMapper, mirroring what main.go does with real clients.
func newFakeProvisioner(t *testing.T) (*Provisioner, *dynamicfake.FakeDynamicClient) {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := networkingv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	mapper := meta.NewDefaultRESTMapper(nil)
	gvks := []schema.GroupVersionKind{
		{Version: "v1", Kind: "LimitRange"},
		{Version: "v1", Kind: "ResourceQuota"},
		{Version: "v1", Kind: "ServiceAccount"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"},
		{Group: "networking.k8s.io", Version: "v1", Kind: "NetworkPolicy"},
	}
	for _, gvk := range gvks {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}

	client := dynamicfake.NewSimpleDynamicClient(scheme)
	prov := New(client, mapper, config.NewDefaultConfig(), discardLogger)
	return prov, client
}

func TestNewStaticMapper(t *testing.T) {
	cfg := config.NewDefaultConfig()
	mapper := NewStaticMapper(cfg)
	want := map[schema.GroupVersionKind]schema.GroupVersionResource{
		{Version: "v1", Kind: "LimitRange"}:                                      {Version: "v1", Resource: "limitranges"},
		{Version: "v1", Kind: "ResourceQuota"}:                                   {Version: "v1", Resource: "resourcequotas"},
		{Version: "v1", Kind: "ServiceAccount"}:                                  {Version: "v1", Resource: "serviceaccounts"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"}:        {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "roles"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"}: {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "rolebindings"},
		{Group: "networking.k8s.io", Version: "v1", Kind: "NetworkPolicy"}:       {Group: "networking.k8s.io", Version: "v1", Resource: "networkpolicies"},
	}
	for gvk, wantGVR := range want {
		got, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			t.Errorf("RESTMapping(%s): %v", gvk, err)
			continue
		}
		if got.Resource != wantGVR {
			t.Errorf("RESTMapping(%s).Resource = %s, want %s", gvk, got.Resource, wantGVR)
		}
	}
	if n := len(GVKs(cfg)); n != len(want) {
		t.Errorf("GVKs returned %d kinds, want %d", n, len(want))
	}
}

func gvrFor(t *testing.T, mapper meta.RESTMapper, obj interface {
	GroupVersionKind() schema.GroupVersionKind
}) schema.GroupVersionResource {
	t.Helper()
	gvk := obj.GroupVersionKind()
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		t.Fatalf("rest mapping for %s: %v", gvk, err)
	}
	return mapping.Resource
}

func TestProvisionCreatesAllResources(t *testing.T) {
	prov, client := newFakeProvisioner(t)
	ctx := context.Background()
	ns := "production"

	if err := prov.Provision(ctx, ns); err != nil {
		t.Fatalf("provision: %v", err)
	}

	// Re-resolve the mapper from the provisioner's objects and confirm each
	// object now exists in the (fake) cluster.
	mapper := newProbeMapper()
	cfg := config.NewDefaultConfig()
	for _, obj := range Objects(ns, cfg) {
		gvr := gvrFor(t, mapper, obj)
		got, err := client.Resource(gvr).Namespace(ns).Get(ctx, obj.GetName(), metav1.GetOptions{})
		if err != nil {
			t.Errorf("get %s/%s: %v", obj.GetKind(), obj.GetName(), err)
			continue
		}
		if got.GetName() != obj.GetName() {
			t.Errorf("got name %q, want %q", got.GetName(), obj.GetName())
		}
	}
}

func TestProvisionIsIdempotent(t *testing.T) {
	prov, client := newFakeProvisioner(t)
	ctx := context.Background()
	ns := "production"

	// First provision creates everything.
	if err := prov.Provision(ctx, ns); err != nil {
		t.Fatalf("first provision: %v", err)
	}
	// Same provisioner: second call short-circuits via the done set.
	if err := prov.Provision(ctx, ns); err != nil {
		t.Fatalf("second provision (done set): %v", err)
	}

	// A fresh provisioner (empty done set) sharing the same client must still
	// succeed because every object already exists and is skipped.
	fresh := New(client, newProbeMapper(), config.NewDefaultConfig(), discardLogger)
	if err := fresh.Provision(ctx, ns); err != nil {
		t.Fatalf("fresh provisioner (already-exists path): %v", err)
	}
}

func TestMarkDoneSkipsProvisioning(t *testing.T) {
	prov, client := newFakeProvisioner(t)
	ctx := context.Background()
	ns := "existing"

	prov.MarkDone(ns)
	if err := prov.Provision(ctx, ns); err != nil {
		t.Fatalf("provision of marked namespace: %v", err)
	}

	// Nothing should have been created for the marked namespace.
	mapper := newProbeMapper()
	for _, obj := range Objects(ns, config.NewDefaultConfig()) {
		gvr := gvrFor(t, mapper, obj)
		if _, err := client.Resource(gvr).Namespace(ns).Get(ctx, obj.GetName(), metav1.GetOptions{}); err == nil {
			t.Errorf("expected %s/%s to not be created", obj.GetKind(), obj.GetName())
		}
	}
}

// newProbeMapper builds the same static mapper the fake provisioner uses, for
// independent lookup in tests.
func newProbeMapper() meta.RESTMapper {
	mapper := meta.NewDefaultRESTMapper(nil)
	gvks := []schema.GroupVersionKind{
		{Version: "v1", Kind: "LimitRange"},
		{Version: "v1", Kind: "ResourceQuota"},
		{Version: "v1", Kind: "ServiceAccount"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"},
		{Group: "networking.k8s.io", Version: "v1", Kind: "NetworkPolicy"},
	}
	for _, gvk := range gvks {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}
	return mapper
}
