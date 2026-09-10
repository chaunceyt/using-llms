package handler

import (
	"encoding/json"
	"net/http"
)

// ItemHandler handles HTTP requests for the /items resource.
type ItemHandler struct {
	deps Deps
}

// NewItemHandler returns an ItemHandler backed by the given dependencies.
func NewItemHandler(deps Deps) *ItemHandler {
	return &ItemHandler{deps: deps}
}

type createItemRequest struct {
	Name string `json:"name"`
}

// ListItems handles GET /items.
func (h *ItemHandler) ListItems(w http.ResponseWriter, r *http.Request) {
	items, err := h.deps.Items.List(r.Context())
	if err != nil {
		writeServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, items)
}

// GetItem handles GET /items/{id}.
func (h *ItemHandler) GetItem(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	item, err := h.deps.Items.Get(r.Context(), id)
	if err != nil {
		writeServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, item)
}

// CreateItem handles POST /items.
func (h *ItemHandler) CreateItem(w http.ResponseWriter, r *http.Request) {
	var req createItemRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	item, err := h.deps.Items.Create(r.Context(), req.Name)
	if err != nil {
		writeServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, item)
}

// DeleteItem handles DELETE /items/{id}.
func (h *ItemHandler) DeleteItem(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.deps.Items.Delete(r.Context(), id); err != nil {
		writeServiceError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
