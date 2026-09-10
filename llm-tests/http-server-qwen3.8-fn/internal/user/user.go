// Package user defines the user domain model and its storage contract.
package user

import (
	"context"
	"errors"
	"time"
)

var ErrNotFound = errors.New("user not found")

type User struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	CreatedAt time.Time `json:"created_at"`
}

// Store is the persistence contract for users. Handlers depend on this
// interface, not on a concrete implementation.
type Store interface {
	Get(ctx context.Context, id string) (User, error)
	List(ctx context.Context) ([]User, error)
	Create(ctx context.Context, u User) (User, error)
	Delete(ctx context.Context, id string) error
}
