package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"example.com/httpserver/internal/config"
	"example.com/httpserver/internal/handler"
	"example.com/httpserver/internal/repository"
	"example.com/httpserver/internal/service"
)

// newTestHandler builds a fully-wired router for testing, exercising the whole
// stack: router, middleware, handlers, service, and in-memory repository.
func newTestHandler(t *testing.T) http.Handler {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := config.Default()
	cfg.Environment = "test"

	repo := repository.NewMemoryItemRepository()
	deps := handler.NewDeps(service.NewItemService(repo), &cfg)
	return NewRouter(deps, logger)
}

func do(t *testing.T, h http.Handler, method, path string, body io.Reader) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, body)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestLiveness(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodGet, "/healthz", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if rec.Header().Get("X-Request-Id") == "" {
		t.Error("expected X-Request-Id response header")
	}
}

func TestReadiness(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodGet, "/readyz", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
}

func TestVersion(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodGet, "/version", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	var body map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["environment"] != "test" {
		t.Errorf("environment = %q, want test", body["environment"])
	}
}

func TestCreateItemThenGet(t *testing.T) {
	h := newTestHandler(t)

	rec := do(t, h, http.MethodPost, "/items", bytes.NewBufferString(`{"name":"widget"}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create status = %d, want %d", rec.Code, http.StatusCreated)
	}
	var item struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&item); err != nil {
		t.Fatalf("decode created item: %v", err)
	}
	if item.ID == "" {
		t.Fatal("expected non-empty ID in create response")
	}

	rec = do(t, h, http.MethodGet, "/items/"+item.ID, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("get status = %d, want %d", rec.Code, http.StatusOK)
	}
	var got map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatalf("decode get item: %v", err)
	}
	if got["name"] != "widget" {
		t.Errorf("name = %v, want widget", got["name"])
	}
}

func TestListItems(t *testing.T) {
	h := newTestHandler(t)

	for _, n := range []string{"a", "b"} {
		rec := do(t, h, http.MethodPost, "/items", bytes.NewBufferString(`{"name":"`+n+`"}`))
		if rec.Code != http.StatusCreated {
			t.Fatalf("create status = %d, want %d", rec.Code, http.StatusCreated)
		}
	}

	rec := do(t, h, http.MethodGet, "/items", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("list status = %d, want %d", rec.Code, http.StatusOK)
	}
	var items []map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&items); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(items) != 2 {
		t.Errorf("list len = %d, want 2", len(items))
	}
}

func TestGetItemNotFound(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodGet, "/items/missing", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestDeleteItem(t *testing.T) {
	h := newTestHandler(t)

	rec := do(t, h, http.MethodPost, "/items", bytes.NewBufferString(`{"name":"widget"}`))
	var item struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&item); err != nil {
		t.Fatalf("decode: %v", err)
	}

	rec = do(t, h, http.MethodDelete, "/items/"+item.ID, nil)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete status = %d, want %d", rec.Code, http.StatusNoContent)
	}

	rec = do(t, h, http.MethodGet, "/items/"+item.ID, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("get after delete status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestDeleteItemNotFound(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodDelete, "/items/missing", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestCreateItemInvalidBody(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodPost, "/items", bytes.NewBufferString(`{not json`))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestCreateItemEmptyName(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodPost, "/items", bytes.NewBufferString(`{"name":""}`))
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnprocessableEntity)
	}
}

func TestRootIndex(t *testing.T) {
	rec := do(t, newTestHandler(t), http.MethodGet, "/", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := body["endpoints"]; !ok {
		t.Error("expected endpoints in root response")
	}
}

func TestMethodNotAllowed(t *testing.T) {
	// A path with no PUT route should yield 405 from the method-aware mux.
	rec := do(t, newTestHandler(t), http.MethodPut, "/items", nil)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}
