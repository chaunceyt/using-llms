// Package repository defines persistence operations for the application.
package repository

import (
	"context"
	"errors"
	"sort"
	"strconv"
	"sync"
	"time"

	"example.com/httpserver/internal/models"
)

// ErrNotFound is returned when a requested item does not exist.
var ErrNotFound = errors.New("item not found")

// ItemRepository defines the persistence operations for items.
type ItemRepository interface {
	GetAll(ctx context.Context) ([]models.Item, error)
	GetByID(ctx context.Context, id string) (models.Item, error)
	Create(ctx context.Context, name string) (models.Item, error)
	Delete(ctx context.Context, id string) error
}

// memoryItemRepository is a concurrency-safe, in-memory ItemRepository. It
// exists so the application is self-contained; a production system would
// swap in a database-backed implementation of the same interface.
type memoryItemRepository struct {
	mu    sync.RWMutex
	items map[string]models.Item
	seq   int
}

// NewMemoryItemRepository returns an in-memory ItemRepository.
func NewMemoryItemRepository() ItemRepository {
	return &memoryItemRepository{
		items: make(map[string]models.Item),
	}
}

func (r *memoryItemRepository) GetAll(_ context.Context) ([]models.Item, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]models.Item, 0, len(r.items))
	for _, it := range r.items {
		out = append(out, it)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func (r *memoryItemRepository) GetByID(_ context.Context, id string) (models.Item, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	it, ok := r.items[id]
	if !ok {
		return models.Item{}, ErrNotFound
	}
	return it, nil
}

func (r *memoryItemRepository) Create(_ context.Context, name string) (models.Item, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.seq++
	it := models.Item{
		ID:        strconv.Itoa(r.seq),
		Name:      name,
		CreatedAt: time.Now().UTC(),
	}
	r.items[it.ID] = it
	return it, nil
}

func (r *memoryItemRepository) Delete(_ context.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.items[id]; !ok {
		return ErrNotFound
	}
	delete(r.items, id)
	return nil
}
