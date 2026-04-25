package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/hybridgroup/yzma/pkg/llama"
)

var (
	modelFile   string
	libPath     string
	verbose     bool
	temperature float64
	predictSize int
	promptFlag  string
)

func main() {
	// Parse flags
	defaultModel := fmt.Sprintf("%s/%s", "static", "gemma-4-E4B-it-Q8_0.gguf")

	flag.StringVar(&modelFile, "model", defaultModel, "path to GGUF model file")
	flag.StringVar(&libPath, "lib", os.Getenv("YZMA_LIB"), "path to llama.cpp compiled library (or set YZMA_LIB)")
	flag.BoolVar(&verbose, "v", false, "verbose logging")
	flag.Float64Var(&temperature, "temperature", 0.5, "prediction temperature")
	flag.IntVar(&predictSize, "n", 1024, "max tokens to predict per generation")
	flag.StringVar(&promptFlag, "prompt", "", "prompt to use directly (skip the TUI form)")
	flag.Parse()

	if libPath == "" {
		slog.Error("Error: provide -lib flag or set YZMA_LIB environment variable")
		os.Exit(1)
	}

	// Load llama.cpp library and initialize
	if err := llama.Load(libPath); err != nil {
		slog.Error("Failed to load llama library", "err", err)
		os.Exit(1)
	}

	if !verbose {
		llama.LogSet(llama.LogSilent())
	}

	// Initialize
	llama.Init()
	defer llama.Close()

	// Load model from model file
	loadStart := time.Now()
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		slog.Error("Failed to load model", "err", err)
		os.Exit(1)
	}
	defer llama.ModelFree(model)
	slog.Info("model loaded", "model", modelFile, "elapsed", time.Since(loadStart).Round(time.Millisecond))

	// Create inference context. Set explicit NCtx and match NBatch/NUbatch
	// to it so any prompt that fits in context can be decoded in a single
	// batch — llama.cpp asserts if the prompt exceeds NBatch.
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = ctxParams.NCtx
	ctxParams.NUbatch = ctxParams.NCtx

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		slog.Error("Failed to create context", "err", err)
		os.Exit(1)
	}
	defer llama.Free(ctx)

	// Load model vocabulary. We don't read the embedded chat template — the
	// app uses its own renderGemmaPrompt (see prompting.go) for formatting.
	vocab := llama.ModelGetVocab(model)

	// Create sampler for model predictions.
	sp := llama.DefaultSamplerParams()
	sp.Temp = float32(temperature)
	sampler := llama.NewSampler(model, llama.DefaultSamplers, sp)

	defer llama.SamplerFree(sampler)

	// Get tool definitions
	tools := getToolDefinitions()

	// Get user prompt: from -prompt flag (for scripting/benchmarking) or
	// from the TUI form (interactive use).
	var prompt string
	if promptFlag != "" {
		prompt = promptFlag
	} else {
		var err error
		prompt, err = createForm()
		if err != nil {
			slog.Error("Failed to get user prompt", "err", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Asking: %s\n\n", prompt)

	err = spinner.New().
		Title("Getting legislator info ...").
		Action(func() {
			if err := runConversation(ctx, vocab, sampler, tools, prompt); err != nil {
				slog.Error("Conversation failed", "err", err)
				os.Exit(1)
			}
		}).Run()
	if err != nil {
		slog.Error("Spinner error", "err", err)
	}
}

// Create TUI form to get user prompt
func createForm() (string, error) {
	var prompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("US Legislators (powered by Gemma + congress-legislators)").
				Description("Ask about a US Legislator (Ctrl+C to cancel)"),

			huh.NewInput().
				Title("Ask about any US legislator").
				Placeholder("e.g. Tell me about Chuck Schumer").
				Value(&prompt).
				Validate(func(str string) error {
					if str == "" {
						return fmt.Errorf("please enter a legislator related question")
					}
					return nil
				}),
		),
	)

	if err := form.Run(); err != nil {
		return "", fmt.Errorf("running form: %w", err)
	}

	return prompt, nil
}

func runConversation(
	ctx llama.Context,
	vocab llama.Vocab,
	sampler llama.Sampler,
	tools []Tool,
	userPrompt string,
) error {
	systemPrompt := getSystemPrompt()
	toolDeclarations := renderToolDeclarations(tools)

	// Pass 1: generate with initial prompt (system + user + open model turn)
	prompt := renderGemmaPrompt(systemPrompt, toolDeclarations, userPrompt)
	if verbose {
		fmt.Printf("\n=== Pass 1 Prompt (%d chars) ===\n%s\n===========================\n", len(prompt), prompt)
	}
	response := generateFromPrompt(ctx, vocab, sampler, prompt)
	if verbose {
		fmt.Printf("\n=== Pass 1 Response ===\n%s\n=============================\n", response)
	}

	// Parse native Gemma 4 tool call: <|tool_call>call:name{args}<tool_call|>
	call, found := parseGemma4ToolCall(response)

	// No tool call — model answered directly. Render and return.
	if !found {
		if strings.Contains(response, "<|tool_call>") {
			return fmt.Errorf("model emitted a tool call but parsing failed: %s", response)
		}
		renderMarkdown(response)
		return nil
	}

	// Execute the tool call
	result, err := executeToolCall(call)
	if err != nil {
		result = fmt.Sprintf("Error: %v", err)
	}
	if verbose {
		fmt.Printf("Tool call: %s(%v) => %s\n", call.Function.Name, call.Function.Arguments, result[:min(len(result), 200)])
	}

	// Use the model's raw response as the tool call text for pass 2,
	// and format the tool response in Gemma 4 native tokens.
	toolCallText := strings.TrimSpace(response)
	toolResponseText := fmt.Sprintf(
		"<|tool_response>response:%s{%s}<tool_response|>\n",
		call.Function.Name, result)

	// Pass 2: generate with tool result injected into the same model turn
	prompt = renderGemmaPromptWithToolResult(
		systemPrompt, toolDeclarations, userPrompt,
		toolCallText, toolResponseText)
	if verbose {
		fmt.Printf("\n=== Pass 2 Prompt (%d chars) ===\n%s\n===========================\n", len(prompt), prompt)
	}
	response = generateFromPrompt(ctx, vocab, sampler, prompt)
	if verbose {
		fmt.Printf("\n=== Pass 2 Response ===\n%s\n=============================\n", response)
	}

	renderMarkdown(response)
	return nil
}

// generateFromPrompt clears the KV cache, tokenizes a prompt, and runs
// inference. Wraps the clear → tokenize → generate sequence that both
// passes of runConversation need.
func generateFromPrompt(ctx llama.Context, vocab llama.Vocab, sampler llama.Sampler, prompt string) string {
	mem, _ := llama.GetMemory(ctx)
	llama.MemoryClear(mem, true)

	// addSpecial=true lets the tokenizer prepend BOS if the model's
	// metadata requires it; parseSpecial=true converts the literal
	// Gemma 4 control tokens (<|turn>, <turn|>, <|tool>, etc.) in our
	// rendered prompt into their special token IDs.
	tokens := llama.Tokenize(vocab, prompt, true, true)
	return generate(ctx, vocab, sampler, tokens)
}

// renderMarkdown renders a response string as styled Markdown in a bordered box.
func renderMarkdown(text string) {
	out, err := glamour.Render(text, "dark")
	if err != nil {
		fmt.Println(text)
		return
	}

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("63")).
		Padding(0, 1)

	fmt.Println(box.Render(strings.TrimSpace(out)))
}

// generate runs token-by-token inference for up to predictSize generated tokens
// or until the model emits an end-of-generation token, then returns the decoded
// response.
func generate(ctx llama.Context, vocab llama.Vocab, sampler llama.Sampler, tokens []llama.Token) string {
	var response strings.Builder
	// Pre-allocate the piece buffer once and reuse it across iterations to avoid
	// allocating 256 bytes of garbage per generated token.
	buf := make([]byte, 256)

	// Decode the entire prompt in one batch (prefill).
	batch := llama.BatchGetOne(tokens)
	llama.Decode(ctx, batch)

	// Generation loop: iterate up to predictSize times. Each iteration samples
	// one token, checks for EOG, decodes it, and prepares the next batch.
	for i := 0; i < predictSize; i++ {
		token := llama.SamplerSample(sampler, ctx, -1)
		if llama.VocabIsEOG(vocab, token) {
			break
		}

		// special=false: skip rendering any special tokens (e.g. Gemma's
		// <end_of_turn>, <start_of_turn>, or other control markers) that the
		// model might emit before its true EOG token. With special=true those
		// would leak into the response as visible "<...>" text. We rely on
		// VocabIsEOG above to terminate generation correctly, and on the
		// model's <tool_call> being plain text (not a special vocab entry)
		// for tool-call parsing to still work.
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		response.Write(buf[:n])

		batch = llama.BatchGetOne([]llama.Token{token})
		llama.Decode(ctx, batch)
	}

	return strings.TrimSpace(response.String())
}
