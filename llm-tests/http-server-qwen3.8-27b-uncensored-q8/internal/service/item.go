// Package service contains the application's business logic, sitting between
// the HTTP handlers and the persistence layer.
package service

import (
	"context"
	"errors"
	"strings"

	"example.com/httpserver/internal/models"
	"example.com/httpserver/internal/repository"
)

// Sentinel errors returned by the service. Handlers map these to HTTP status
// codes.
var (
	// ErrNotFound indicates the requested item does not exist.
	ErrNotFound = errors.New("item not found")
	// ErrInvalidName indicates the item name failed validation.
	ErrInvalidName = errors.New("invalid item name")
)

// ItemService implements the business logic for managing items.
type ItemService struct {
	repo repository.ItemRepository
}

// NewItemService returns an ItemService backed by the given repository.
func NewItemService(repo repository.ItemRepository) *ItemService {
	return &ItemService{repo: repo}
}

// List returns all items.
func (s *ItemService) List(ctx context.Context) ([]models.Item, error) {
	return s.repo.GetAll(ctx)
}

// Get returns the item with the given ID.
func (s *ItemService) Get(ctx context.Context, id string) (models.Item, error) {
	it, err := s.repo.GetByID(ctx, id)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return models.Item{}, ErrNotFound
		}
		return models.Item{}, err
	}
	return it, nil
}

// Create validates the name and stores a new item.
func (s *ItemService) Create(ctx context.Context, name string) (models.Item, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return models.Item{}, ErrInvalidName
	}
	return s.repo.Create(ctx, name)
}

// Delete removes the item with the given ID.
func (s *ItemService) Delete(ctx context.Context, id string) error {
	if err := s.repo.Delete(ctx, id); err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return ErrNotFound
		}
		return err
	}
	return nil
}
