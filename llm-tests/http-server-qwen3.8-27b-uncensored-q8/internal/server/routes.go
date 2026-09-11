package server

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"example.com/httpserver/internal/handler"
	"example.com/httpserver/internal/middleware"
)

// NewRouter builds the application's root http.Handler with all routes and
// the shared middleware chain.
func NewRouter(deps handler.Deps, logger *slog.Logger) http.Handler {
	items := handler.NewItemHandler(deps)
	health := handler.NewHealthHandler(deps)

	mux := http.NewServeMux()

	// Service health & metadata.
	mux.HandleFunc("GET /healthz", health.Liveness)
	mux.HandleFunc("GET /readyz", health.Readiness)
	mux.HandleFunc("GET /version", health.Version)

	// Items resource.
	mux.HandleFunc("GET /items", items.ListItems)
	mux.HandleFunc("POST /items", items.CreateItem)
	mux.HandleFunc("GET /items/{id}", items.GetItem)
	mux.HandleFunc("DELETE /items/{id}", items.DeleteItem)

	// Root index listing available endpoints.
	mux.HandleFunc("GET /{$}", rootIndex)

	// Middleware order matters: RequestID is outermost so the ID is available
	// to Logging; Recovery is innermost so it catches panics from handlers and
	// lets Logging record the resulting status.
	return middleware.Chain(mux,
		middleware.RequestID,
		middleware.Logging(logger),
		middleware.Recovery(logger),
	)
}

func rootIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"service": "httpserver",
		"endpoints": []string{
			"GET    /healthz",
			"GET    /readyz",
			"GET    /version",
			"GET    /items",
			"POST   /items",
			"GET    /items/{id}",
			"DELETE /items/{id}",
		},
	})
}
