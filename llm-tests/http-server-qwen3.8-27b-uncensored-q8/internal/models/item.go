// Package models defines the domain entities used across the application.
package models

import "time"

// Item is the domain entity served by the API.
type Item struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
}
