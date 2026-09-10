package handlers

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealth(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		wantCode int
	}{
		{name: "health", path: "/healthz", wantCode: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			rec := httptest.NewRecorder()

			Health().ServeHTTP(rec, req)

			if rec.Code != tt.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tt.wantCode)
			}

			var body map[string]string
			if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			if body["status"] != "ok" {
				t.Errorf(`body status = %q, want "ok"`, body["status"])
			}
		})
	}
}

func TestReady(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rec := httptest.NewRecorder()

	Ready().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}

	var body map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body["status"] != "ready" {
		t.Errorf(`body status = %q, want "ready"`, body["status"])
	}
}

func TestGreeting(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		wantCode int
		wantName string
	}{
		{name: "default name", query: "", wantCode: http.StatusOK, wantName: "world"},
		{name: "custom name", query: "?name=Ada", wantCode: http.StatusOK, wantName: "Ada"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// NewAPI is mounted behind StripPrefix("/api/v1") in server.go, so
			// its own routes are relative to that prefix.
			req := httptest.NewRequest(http.MethodGet, "/hello"+tt.query, nil)
			rec := httptest.NewRecorder()

			mux := NewAPI(testLogger())
			// Route through the ServeMux so path-based routing is exercised.
			mux.ServeHTTP(rec, req)

			if rec.Code != tt.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tt.wantCode)
			}

			var body Greeting
			if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			if body.Name != tt.wantName {
				t.Errorf(`name = %q, want %q`, body.Name, tt.wantName)
			}
			if body.Message == "" {
				t.Error("message is empty")
			}
		})
	}
}

// TestWriteJSONRejectsTrailingSlashMethod guards the ServeMux routing contract.
func TestAPIMethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/hello", nil)
	rec := httptest.NewRecorder()

	mux := NewAPI(testLogger())
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

// testLogger returns a slog logger that writes nowhere, keeping test output
// quiet while exercising the real handler code path.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
