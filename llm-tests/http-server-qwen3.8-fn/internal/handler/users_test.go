package handler

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

func newTestUsers() *Users {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewUsers(user.NewMemoryStore(), logger)
}

func TestUsers_Create(t *testing.T) {
	h := newTestUsers()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/users",
		strings.NewReader(`{"name":"Ada","email":"ada@example.com"}`))
	rec := httptest.NewRecorder()

	h.Create(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusCreated)
	}
	if loc := rec.Header().Get("Location"); !strings.HasPrefix(loc, "/api/v1/users/") {
		t.Errorf("Location = %q", loc)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type = %q", ct)
	}
}

func TestUsers_Create_InvalidEmail(t *testing.T) {
	h := newTestUsers()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/users",
		strings.NewReader(`{"name":"Ada","email":"not-an-email"}`))
	rec := httptest.NewRecorder()

	h.Create(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnprocessableEntity)
	}
}

func TestUsers_Create_RejectsUnknownFields(t *testing.T) {
	h := newTestUsers()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/users",
		strings.NewReader(`{"name":"Ada","email":"ada@example.com","admin":true}`))
	rec := httptest.NewRecorder()

	h.Create(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestUsers_Create_RejectsOversizedBody(t *testing.T) {
	h := newTestUsers()
	body := `{"name":"` + strings.Repeat("x", maxBodyBytes+1) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/users", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Create(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestUsers_Get_NotFound(t *testing.T) {
	h := newTestUsers()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/users/missing", nil)
	req.SetPathValue("id", "missing")
	rec := httptest.NewRecorder()

	h.Get(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}
