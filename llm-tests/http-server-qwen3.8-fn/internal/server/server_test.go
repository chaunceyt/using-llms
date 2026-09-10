package server

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

func testHandler() http.Handler {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return Chain(newMux(user.NewMemoryStore(), logger), requestID, recoverer(logger))
}

func TestRoutes(t *testing.T) {
	h := testHandler()

	tests := []struct {
		method, path string
		want         int
	}{
		{http.MethodGet, "/healthz", http.StatusOK},
		{http.MethodGet, "/readyz", http.StatusOK},
		{http.MethodGet, "/api/v1/users", http.StatusOK},
		{http.MethodGet, "/api/v1/users/abc", http.StatusNotFound},
		{http.MethodDelete, "/api/v1/users", http.StatusNotFound},
		{http.MethodGet, "/nope", http.StatusNotFound},
	}
	for _, tt := range tests {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(tt.method, tt.path, nil))
		if rec.Code != tt.want {
			t.Errorf("%s %s: status = %d, want %d", tt.method, tt.path, rec.Code, tt.want)
		}
	}
}

func TestRequestIDIsSet(t *testing.T) {
	rec := httptest.NewRecorder()
	testHandler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if rec.Header().Get("X-Request-ID") == "" {
		t.Error("expected X-Request-ID header to be set")
	}
}
