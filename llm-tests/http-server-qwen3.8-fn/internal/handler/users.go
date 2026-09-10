package handler

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/mail"
	"strings"

	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

const maxBodyBytes = 1 << 20 // 1 MiB

type Users struct {
	store  user.Store
	logger *slog.Logger
}

func NewUsers(store user.Store, logger *slog.Logger) *Users {
	return &Users{store: store, logger: logger}
}

type createUserRequest struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

func (h *Users) List(w http.ResponseWriter, r *http.Request) {
	users, err := h.store.List(r.Context())
	if err != nil {
		writeInternalError(w, r, h.logger, err)
		return
	}
	writeJSON(w, h.logger, http.StatusOK, users)
}

func (h *Users) Get(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	u, err := h.store.Get(r.Context(), id)
	if err != nil {
		mapStoreError(w, r, h.logger, err)
		return
	}
	writeJSON(w, h.logger, http.StatusOK, u)
}

func (h *Users) Create(w http.ResponseWriter, r *http.Request) {
	var req createUserRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, h.logger, http.StatusBadRequest, err.Error())
		return
	}
	if msg := validate(req); msg != "" {
		writeError(w, h.logger, http.StatusUnprocessableEntity, msg)
		return
	}

	u, err := h.store.Create(r.Context(), user.User{
		Name:  strings.TrimSpace(req.Name),
		Email: strings.ToLower(strings.TrimSpace(req.Email)),
	})
	if err != nil {
		writeInternalError(w, r, h.logger, err)
		return
	}
	w.Header().Set("Location", "/api/v1/users/"+u.ID)
	writeJSON(w, h.logger, http.StatusCreated, u)
}

func (h *Users) Delete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.store.Delete(r.Context(), id); err != nil {
		mapStoreError(w, r, h.logger, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var maxErr *http.MaxBytesError
		switch {
		case errors.Is(err, io.EOF):
			return errors.New("request body must not be empty")
		case errors.As(err, &maxErr):
			return errors.New("request body too large")
		default:
			return errors.New("request body must be valid JSON")
		}
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("request body must contain a single JSON object")
	}
	return nil
}

func validate(req createUserRequest) string {
	if strings.TrimSpace(req.Name) == "" {
		return "name is required"
	}
	if _, err := mail.ParseAddress(req.Email); err != nil {
		return "email must be a valid address"
	}
	return ""
}
