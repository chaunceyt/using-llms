package main

import (
	"encoding/json"
	"fmt"

	"github.com/hybridgroup/yzma/pkg/message"
)

// Tool represents a tool definition for the LLM.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents a function definition.
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

func getToolDefinitions() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_legislator",
				Description: "Get the contact information for a US legislator by fullname",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"fullname": map[string]interface{}{
							"type":        "string",
							"description": "Legislator's fullname, e.g. 'Chuck Schumer'",
						},
					},
					"required": []string{"fullname"},
				},
			},
		},
	}
}

// executeToolCall executes a tool call and returns the result.
func executeToolCall(call message.ToolCall) (string, error) {
	switch call.Function.Name {
	case "get_legislator":
		fullname, ok := call.Function.Arguments["fullname"]
		if !ok || fullname == "" {
			return "", fmt.Errorf("missing 'fullname' argument")
		}
		legislator, err := GetLegislator(fullname)
		if err != nil {
			return "", fmt.Errorf("legislator lookup failed: %w", err)
		}
		result, err := json.Marshal(legislator)
		if err != nil {
			return "", fmt.Errorf("marshal legislator: %w", err)
		}
		return string(result), nil
	default:
		return "", fmt.Errorf("unknown function: %s", call.Function.Name)
	}
}
