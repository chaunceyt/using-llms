package middleware

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

func silentLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestRequestIDGeneratesWhenAbsent(t *testing.T) {
	var saw string
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		saw, _ = RequestIDFromContext(r.Context())
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if saw == "" {
		t.Error("expected a request ID in the context")
	}
	if rec.Header().Get(RequestIDHeader) != saw {
		t.Errorf("response header = %q, want %q", rec.Header().Get(RequestIDHeader), saw)
	}
}

func TestRequestIDPreservesIncoming(t *testing.T) {
	var saw string
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		saw, _ = RequestIDFromContext(r.Context())
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set(RequestIDHeader, "fixed-id")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if saw != "fixed-id" {
		t.Errorf("context request ID = %q, want fixed-id", saw)
	}
	if rec.Header().Get(RequestIDHeader) != "fixed-id" {
		t.Errorf("response header = %q, want fixed-id", rec.Header().Get(RequestIDHeader))
	}
}

func TestRecoveryConvertsPanicTo500(t *testing.T) {
	h := Recovery(silentLogger())(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("boom")
	}))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusInternalServerError)
	}
}

func TestRecoveryPassesThroughOnSuccess(t *testing.T) {
	h := Recovery(silentLogger())(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
}

func TestLoggingCapturesStatus(t *testing.T) {
	called := false
	h := Logging(silentLogger())(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusTeapot)
	}))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if !called {
		t.Fatal("inner handler was not called")
	}
	if rec.Code != http.StatusTeapot {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusTeapot)
	}
}

func TestChainOrdering(t *testing.T) {
	var order []string
	h := Chain(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}),
		func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				order = append(order, "A")
				next.ServeHTTP(w, r)
			})
		},
		func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				order = append(order, "B")
				next.ServeHTTP(w, r)
			})
		},
	)

	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))

	if len(order) != 2 || order[0] != "A" || order[1] != "B" {
		t.Errorf("middleware order = %v, want [A B]", order)
	}
}
