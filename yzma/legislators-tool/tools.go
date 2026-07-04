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
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "web_search",
				Description: "Search the web for a US legislator by fullname",
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
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "web_read",
				Description: "Read the content for the URL provided about a legislator",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"url": map[string]interface{}{
							"type":        "string",
							"description": "A URL from web search results about a legislator",
						},
					},
					"required": []string{"url"},
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
	case "web_search":
		fullname, ok := call.Function.Arguments["fullname"]
		if !ok || fullname == "" {
			return "", fmt.Errorf("missing 'fullname' argument")
		}

		search, err := searchQuery(fullname)
		if err != nil {
			return "", fmt.Errorf("marshal legislator: %w", err)
		}
		result, err := json.Marshal(search)
		if err != nil {
			return "", fmt.Errorf("marshal legislator: %w", err)
		}
		return string(result), nil
	case "web_read":
		url, ok := call.Function.Arguments["url"]
		if !ok || url == "" {
			return "", fmt.Errorf("missing 'url' argument")
		}

		search, err := readQuery(url)
		if err != nil {
			return "", fmt.Errorf("marshal legislator: %w", err)
		}
		result, err := json.Marshal(search)
		if err != nil {
			return "", fmt.Errorf("marshal legislator: %w", err)
		}
		return string(result), nil
	default:
		return "", fmt.Errorf("unknown function: %s", call.Function.Name)
	}
}

func searchQuery(query string) (string, error) {
	searchQuery := `
{
    "query": "%s",
	"region": "us-en",
	"time_range": "365"
}`
	queryStr := fmt.Sprintf(searchQuery, query)
	search := webSearch(json.RawMessage(queryStr))
	return search, nil
}

func readQuery(url string) (string, error) {
	readQuery := `
{
	"url": "%s"
}`
	queryStr := fmt.Sprintf(readQuery, url)
	search := webRead(json.RawMessage(queryStr))
	return search, nil
}
