package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/hybridgroup/yzma/pkg/message"
)

const (
	systemPromptTemplate = `
You are a helpful legislator assistant that ONLY handles legislator requests.
If the user asks something unrelated to legistlator, politely let them know that this application only handles legistlor requests.
Use the get_legislator tool to fetch real congress legislators data, then provide a clear,
well-formatted summary as a wikipedia style page.

Include all data from get_legislator result in to the summary.
`
)

// getSystemPrompt returns the system prompt text (without tool definitions —
// those are rendered separately using native <|tool> tokens).
func getSystemPrompt() string {
	return strings.TrimSpace(systemPromptTemplate)
}

// renderToolDeclarations formats tool definitions using Gemma 4's native
// <|tool>declaration:name{schema}<tool|> syntax.
func renderToolDeclarations(tools []Tool) string {
	var b strings.Builder
	for _, t := range tools {
		schema, _ := json.Marshal(t.Function.Parameters)
		fmt.Fprintf(&b, "\n<|tool>declaration:%s{\"description\":%q,\"parameters\":%s}<tool|>",
			t.Function.Name, t.Function.Description, string(schema))
	}
	return b.String()
}

var (
	// Step 1: extract content between <|tool_call> and <tool_call|>
	// (?s) enables dot-all mode so . matches newlines — the model may
	// wrap long argument values across lines.
	toolCallRe = regexp.MustCompile(`(?s)<\|tool_call>(.+?)<tool_call\|>`)
	// Step 2: extract tool name and JSON-like object from "call:name{...}"
	callBodyRe = regexp.MustCompile(`(?s)^call:(\w+)(\{.+\})$`)
	// Step 3a: quote unquoted JSON keys. Gemma 4's native format uses
	// bare keys (location:) instead of quoted keys ("location":).
	bareKeyRe = regexp.MustCompile(`(\{|,)\s*(\w+)\s*:`)
)

// parseGemma4ToolCall parses a Gemma 4 native tool call from a model response.
// The expected format is:
//
//	<|tool_call>call:function_name{key:<|"|>value<|"|>}<tool_call|>
//
// Parsing proceeds in four steps:
//  1. Extract content from between <|tool_call> and <tool_call|>
//  2. Extract the tool name and JSON-like object from call:name{...}
//  3. Replace <|"|> with " to produce valid JSON
//  4. Unmarshal the JSON into a map[string]string
func parseGemma4ToolCall(response string) (message.ToolCall, bool) {
	// Step 1: extract content between tool_call markers
	m := toolCallRe.FindStringSubmatch(response)
	if m == nil {
		return message.ToolCall{}, false
	}

	// Step 2: extract tool name and JSON-like body
	parts := callBodyRe.FindStringSubmatch(m[1])
	if parts == nil {
		return message.ToolCall{}, false
	}

	// Step 3: convert Gemma 4's native format to valid JSON:
	//   {location:<|"|>New York, NY<|"|>}  →  {"location":"New York, NY"}
	// Replace <|"|> value delimiters with quotes, quote bare keys, and
	// escape any raw newlines inside values (invalid in JSON strings).
	jsonStr := strings.ReplaceAll(parts[2], `<|"|>`, `"`)
	jsonStr = bareKeyRe.ReplaceAllString(jsonStr, `$1"$2":`)
	jsonStr = strings.ReplaceAll(jsonStr, "\n", `\n`)

	// Step 4: unmarshal arguments into ToolFunction
	var args map[string]string
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return message.ToolCall{}, false
	}

	return message.ToolCall{
		Type: "function",
		Function: message.ToolFunction{
			Name:      parts[1],
			Arguments: args,
		},
	}, true
}

// renderGemmaPrompt builds the initial prompt for the first generation pass.
//
// Gemma 4 format (https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4):
//
//	<|turn>system
//	[instructions]
//	<|tool>declaration:name{schema}<tool|><turn|>
//	<|turn>user
//	[question]<turn|>
//	<|turn>model
//
// The literal control tokens (<|turn>, <turn|>, <|tool>, <tool|>) get
// converted to their special token IDs at tokenization time via
// parseSpecial=true in llama.Tokenize.
func renderGemmaPrompt(systemPrompt, toolDeclarations, userPrompt string) string {
	var b strings.Builder
	b.WriteString("<|turn>system\n")
	b.WriteString(systemPrompt)
	b.WriteString(toolDeclarations)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>user\n")
	b.WriteString(userPrompt)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>model\n")
	return b.String()
}

// renderGemmaPromptWithToolResult builds the prompt for the second generation
// pass after a tool call. In Gemma 4's format, the tool call and tool response
// both live inside a single model turn — the turn is left open so the model
// can continue generating the final answer after the tool response.
//
//	<|turn>system
//	[instructions + tool declarations]<turn|>
//	<|turn>user
//	[question]<turn|>
//	<|turn>model
//	<|tool_call>call:name{args}<tool_call|><|tool_response>
//	response:name{result}<tool_response|>
//	[model continues generating here]
func renderGemmaPromptWithToolResult(systemPrompt, toolDeclarations, userPrompt, toolCallText, toolResponseText string) string {
	var b strings.Builder
	b.WriteString("<|turn>system\n")
	b.WriteString(systemPrompt)
	b.WriteString(toolDeclarations)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>user\n")
	b.WriteString(userPrompt)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>model\n")
	b.WriteString(toolCallText)
	b.WriteString(toolResponseText)
	return b.String()
}
