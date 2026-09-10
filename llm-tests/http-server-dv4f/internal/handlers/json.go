package handlers

import (
	"encoding/json"
	"io"
)

// jsonEncode writes value to w as JSON. It returns the error so callers can
// decide how to surface an encode failure.
func jsonEncode(w io.Writer, value any) error {
	return json.NewEncoder(w).Encode(value)
}
