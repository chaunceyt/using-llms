// Package handler contains the HTTP layer: it translates between HTTP
// requests/responses and the service layer.
package handler

import (
	"example.com/httpserver/internal/config"
	"example.com/httpserver/internal/service"
)

// Deps groups the dependencies shared by handlers. Handlers are constructed
// with a Deps value so they can be freely composed and tested.
type Deps struct {
	Items  *service.ItemService
	Config *config.Config
}

// NewDeps returns a Deps value for the given dependencies.
func NewDeps(items *service.ItemService, cfg *config.Config) Deps {
	return Deps{Items: items, Config: cfg}
}
