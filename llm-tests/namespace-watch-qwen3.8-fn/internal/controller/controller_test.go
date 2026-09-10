package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	kubeinformers "k8s.io/client-go/informers"
	kubefake "k8s.io/client-go/kubernetes/fake"

	"namespace-watch-dv4f/internal/defaults"
)

func activeNS(name string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: name, UID: "9a3d-01"},
		Status:     corev1.NamespaceStatus{Phase: corev1.NamespaceActive},
	}
}

func newTestController(t *testing.T, ns *corev1.Namespace) (*Controller, *dynamicfake.FakeDynamicClient, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	factory := kubeinformers.NewSharedInformerFactory(kubefake.NewSimpleClientset(ns), 0)

	listKinds := map[schema.GroupVersionResource]string{}
	for _, r := range defaults.ForNamespace(ns) {
		listKinds[r.GVR] = r.Body.GetKind() + "List"
	}
	dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(), listKinds)
	c := New(dyn, factory, []string{"kube-"})

	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())
	return c, dyn, cancel
}

func TestSyncCreatesAllDefaultsIdempotently(t *testing.T) {
	ns := activeNS("production")
	c, dyn, cancel := newTestController(t, ns)
	defer cancel()
	ctx := context.Background()

	if err := c.sync(ctx, "production"); err != nil {
		t.Fatalf("first sync: %v", err)
	}
	if err := c.sync(ctx, "production"); err != nil {
		t.Fatalf("second sync should tolerate AlreadyExists: %v", err)
	}

	creates := map[schema.GroupVersionResource]int{}
	for _, a := range dyn.Actions() {
		if a.GetVerb() == "create" {
			creates[a.GetResource()]++
		}
	}

	for _, want := range defaults.ForNamespace(ns) {
		if got := creates[want.GVR]; got != 2 {
			t.Errorf("expected 2 create attempts for %s, got %d", want.GVR.Resource, got)
		}

		list, err := dyn.Tracker().List(want.GVR, want.Body.GroupVersionKind(), "production")
		if err != nil {
			t.Fatalf("listing %s: %v", want.GVR.Resource, err)
		}
		ul, ok := list.(*unstructured.UnstructuredList)
		if !ok {
			t.Fatalf("tracker returned %T for %s", list, want.GVR.Resource)
		}
		if len(ul.Items) != 1 {
			t.Errorf("%s: expected exactly 1 object after re-sync, got %d", want.GVR.Resource, len(ul.Items))
			continue
		}
		got := &unstructured.Unstructured{Object: ul.Items[0].Object}
		if got.GetName() != want.Body.GetName() {
			t.Errorf("%s: expected name %q, got %q", want.GVR.Resource, want.Body.GetName(), got.GetName())
		}
		if got.GetNamespace() != "production" {
			t.Errorf("%s: expected namespace production, got %q", want.GVR.Resource, got.GetNamespace())
		}
		refs := got.GetOwnerReferences()
		if len(refs) != 1 || refs[0].Kind != "Namespace" || refs[0].Name != "production" {
			t.Errorf("%s: expected owner reference to Namespace production, got %v", want.GVR.Resource, refs)
		}
	}
}

func TestSyncUnknownNamespaceIsNoop(t *testing.T) {
	ns := activeNS("production")
	c, dyn, cancel := newTestController(t, ns)
	defer cancel()

	if err := c.sync(context.Background(), "does-not-exist"); err != nil {
		t.Fatalf("sync of missing namespace: %v", err)
	}
	for _, a := range dyn.Actions() {
		if a.GetVerb() == "create" {
			t.Fatalf("unexpected create for missing namespace: %v", a)
		}
	}
}

func TestEnqueueSkipsSystemAndInactiveNamespaces(t *testing.T) {
	c, _, cancel := newTestController(t, activeNS("production"))
	defer cancel()

	c.enqueue(activeNS("kube-system"))
	terminating := activeNS("old-app")
	terminating.Status.Phase = corev1.NamespaceTerminating
	c.enqueue(terminating)
	if got := c.queue.Len(); got != 0 {
		t.Fatalf("expected empty queue, got %d items", got)
	}

	c.enqueue(activeNS("team-a"))
	if got := c.queue.Len(); got != 1 {
		t.Fatalf("expected 1 queued namespace, got %d", got)
	}
}
