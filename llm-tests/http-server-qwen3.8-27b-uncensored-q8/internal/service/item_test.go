package service

import (
	"context"
	"errors"
	"testing"

	"example.com/httpserver/internal/repository"
)

func TestCreateTrimsAndValidatesName(t *testing.T) {
	s := NewItemService(repository.NewMemoryItemRepository())
	ctx := context.Background()

	item, err := s.Create(ctx, "  widget  ")
	if err != nil {
		t.Fatalf("Create() error: %v", err)
	}
	if item.Name != "widget" {
		t.Errorf("Name = %q, want trimmed %q", item.Name, "widget")
	}
}

func TestCreateRejectsBlankName(t *testing.T) {
	s := NewItemService(repository.NewMemoryItemRepository())

	for _, name := range []string{"", "   ", "\t"} {
		if _, err := s.Create(context.Background(), name); !errors.Is(err, ErrInvalidName) {
			t.Errorf("Create(%q) error = %v, want ErrInvalidName", name, err)
		}
	}
}

func TestGetMapsNotFound(t *testing.T) {
	s := NewItemService(repository.NewMemoryItemRepository())

	if _, err := s.Get(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get() error = %v, want ErrNotFound", err)
	}
}

func TestDeleteMapsNotFound(t *testing.T) {
	s := NewItemService(repository.NewMemoryItemRepository())

	if err := s.Delete(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Delete() error = %v, want ErrNotFound", err)
	}
}

func TestRoundTrip(t *testing.T) {
	s := NewItemService(repository.NewMemoryItemRepository())
	ctx := context.Background()

	created, err := s.Create(ctx, "widget")
	if err != nil {
		t.Fatalf("Create() error: %v", err)
	}

	got, err := s.Get(ctx, created.ID)
	if err != nil {
		t.Fatalf("Get() error: %v", err)
	}
	if got != created {
		t.Errorf("Get() = %+v, want %+v", got, created)
	}

	list, err := s.List(ctx)
	if err != nil {
		t.Fatalf("List() error: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("List() len = %d, want 1", len(list))
	}

	if err := s.Delete(ctx, created.ID); err != nil {
		t.Fatalf("Delete() error: %v", err)
	}
	if _, err := s.Get(ctx, created.ID); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get() after delete = %v, want ErrNotFound", err)
	}
}
