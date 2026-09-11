package watcher

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	k8sfake "k8s.io/client-go/kubernetes/fake"

	"github.com/example/namespace-watcher/internal/config"
	"github.com/example/namespace-watcher/internal/provisioner"
)

// discardLogger is a logger that swallows output for quiet tests.
var discardLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// newTestProvisioner builds a Provisioner backed by a fake dynamic client so
// provisioning in tests is side-effect free but still exercises the full path.
func newTestProvisioner(t *testing.T) *provisioner.Provisioner {
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
	for _, gvk := range []schema.GroupVersionKind{
		{Version: "v1", Kind: "LimitRange"},
		{Version: "v1", Kind: "ResourceQuota"},
		{Version: "v1", Kind: "ServiceAccount"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
		{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"},
		{Group: "networking.k8s.io", Version: "v1", Kind: "NetworkPolicy"},
	} {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}
	client := dynamicfake.NewSimpleDynamicClient(scheme)
	return provisioner.New(client, mapper, config.NewDefaultConfig(), discardLogger)
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("timed out waiting for condition")
}

func createNamespace(t *testing.T, ctx context.Context, cs *k8sfake.Clientset, name string) {
	t.Helper()
	if _, err := cs.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("create namespace %q: %v", name, err)
	}
}

// runWatcher starts the watcher in a goroutine and registers a cleanup that
// cancels the context and waits for Run to return.
func runWatcher(t *testing.T, w *Watcher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("watcher.Run returned error: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Error("watcher did not shut down within 5s")
		}
	})
	// Give the informer a moment to start syncing.
	time.Sleep(100 * time.Millisecond)
}

func TestWatcherNewOnlyMode(t *testing.T) {
	prov := newTestProvisioner(t)
	cs := k8sfake.NewClientset(
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "existing"}},
	)
	w := New(cs, prov, Options{Backfill: false, Concurrency: 2, Log: discardLogger})
	runWatcher(t, w)

	ctx := context.Background()
	createNamespace(t, ctx, cs, "brand-new")

	waitFor(t, 5*time.Second, func() bool { return prov.IsDone("brand-new") })

	if !prov.IsDone("brand-new") {
		t.Error("expected newly created namespace to be provisioned")
	}
	if prov.IsDone("existing") {
		t.Error("expected pre-existing namespace to NOT be provisioned in new-only mode")
	}
}

func TestWatcherBackfillMode(t *testing.T) {
	prov := newTestProvisioner(t)
	cs := k8sfake.NewClientset(
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "existing"}},
	)
	w := New(cs, prov, Options{Backfill: true, Concurrency: 2, Log: discardLogger})
	runWatcher(t, w)

	// Backfill: the pre-existing namespace must be provisioned on startup.
	waitFor(t, 5*time.Second, func() bool { return prov.IsDone("existing") })

	ctx := context.Background()
	createNamespace(t, ctx, cs, "brand-new")
	waitFor(t, 5*time.Second, func() bool { return prov.IsDone("brand-new") })

	if !prov.IsDone("existing") {
		t.Error("expected pre-existing namespace to be provisioned in backfill mode")
	}
	if !prov.IsDone("brand-new") {
		t.Error("expected newly created namespace to be provisioned in backfill mode")
	}
}
