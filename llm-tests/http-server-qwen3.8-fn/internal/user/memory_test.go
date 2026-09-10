package user

import (
	"context"
	"errors"
	"testing"
)

func TestMemoryStore_CreateGetDelete(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()

	created, err := s.Create(ctx, User{Name: "Ada", Email: "ada@example.com"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if created.ID == "" {
		t.Fatal("Create: expected generated ID")
	}

	got, err := s.Get(ctx, created.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Name != "Ada" {
		t.Errorf("Get: Name = %q, want %q", got.Name, "Ada")
	}

	if err := s.Delete(ctx, created.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	_, err = s.Get(ctx, created.ID)
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("Get after delete: err = %v, want ErrNotFound", err)
	}
}

func TestMemoryStore_List(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()

	for _, name := range []string{"a", "b", "c"} {
		if _, err := s.Create(ctx, User{Name: name, Email: name + "@example.com"}); err != nil {
			t.Fatalf("Create(%s): %v", name, err)
		}
	}

	users, err := s.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(users) != 3 {
		t.Fatalf("List: len = %d, want 3", len(users))
	}
}
