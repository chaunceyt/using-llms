package repository

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestCreateAndGet(t *testing.T) {
	r := NewMemoryItemRepository()
	ctx := context.Background()

	created, err := r.Create(ctx, "widget")
	if err != nil {
		t.Fatalf("Create() error: %v", err)
	}
	if created.ID == "" {
		t.Fatal("Create() returned empty ID")
	}
	if created.Name != "widget" {
		t.Errorf("Name = %q, want widget", created.Name)
	}

	got, err := r.GetByID(ctx, created.ID)
	if err != nil {
		t.Fatalf("GetByID() error: %v", err)
	}
	if got != created {
		t.Errorf("GetByID() = %+v, want %+v", got, created)
	}
}

func TestGetByIDNotFound(t *testing.T) {
	r := NewMemoryItemRepository()

	if _, err := r.GetByID(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("GetByID() error = %v, want ErrNotFound", err)
	}
}

func TestDelete(t *testing.T) {
	r := NewMemoryItemRepository()
	ctx := context.Background()

	created, _ := r.Create(ctx, "widget")
	if err := r.Delete(ctx, created.ID); err != nil {
		t.Fatalf("Delete() error: %v", err)
	}
	if _, err := r.GetByID(ctx, created.ID); !errors.Is(err, ErrNotFound) {
		t.Errorf("GetByID() after delete = %v, want ErrNotFound", err)
	}
}

func TestDeleteNotFound(t *testing.T) {
	r := NewMemoryItemRepository()

	if err := r.Delete(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Delete() error = %v, want ErrNotFound", err)
	}
}

func TestGetAllSorted(t *testing.T) {
	r := NewMemoryItemRepository()
	ctx := context.Background()

	for _, n := range []string{"a", "b", "c"} {
		if _, err := r.Create(ctx, n); err != nil {
			t.Fatalf("Create(%q) error: %v", n, err)
		}
	}

	all, err := r.GetAll(ctx)
	if err != nil {
		t.Fatalf("GetAll() error: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("GetAll() len = %d, want 3", len(all))
	}
	// IDs are numeric strings in creation order, so this must already be sorted.
	for i := 1; i < len(all); i++ {
		if all[i-1].ID > all[i].ID {
			t.Errorf("GetAll() not sorted: %q before %q", all[i-1].ID, all[i].ID)
		}
	}
}

func TestConcurrentAccess(t *testing.T) {
	r := NewMemoryItemRepository()
	ctx := context.Background()

	const n = 100
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			it, err := r.Create(ctx, "x")
			if err != nil {
				t.Errorf("Create() error: %v", err)
				return
			}
			if _, err := r.GetByID(ctx, it.ID); err != nil {
				t.Errorf("GetByID() error: %v", err)
			}
		}()
	}
	wg.Wait()

	all, err := r.GetAll(ctx)
	if err != nil {
		t.Fatalf("GetAll() error: %v", err)
	}
	if len(all) != n {
		t.Errorf("GetAll() len = %d, want %d", len(all), n)
	}
}
