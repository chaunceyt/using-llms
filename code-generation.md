# Generating Code using LLM

## llama-server

### llms used

```
/Volumes/development2/ggufs/gpt-oss-120b-F16.gguf
/Volumes/development2/ggufs/Qwen3-Coder-30B-A3B-Instruct-BF16.gguf
/Volumes/development2/ggufs/Devstral-Small-2507-BF16.gguf
/Volumes/development2/ggufs/MiniMax-M2-Q4_K_M.gguf
/Volumes/development2/ggufs/GLM-4.7-Q4_K_M/GLM-4.7-Q4_K_M-00001-of-00005.gguf
/Volumes/development2/ggufs/mistralai_Devstral-Small-2-24B-Instruct-2512-bf16.gguf
/Volumes/development2/ggufs/GLM-4.5-Air-Q4_K_M.gguf
/Volumes/development3/ggufs/MiniMax-M2.1.q6_k.gguf
/Volumes/development3/ggufs/Devstral-2-123B-Instruct-2512-UD-Q8_K_XL/Devstral-2-123B-Instruct-2512-UD-Q8_K_XL-00001-of-00003.gguf
/Volumes/development3/ggufs/Nemotron-3-Nano-30B-A3B-BF16.gguf
/Volumes/development3/ggufs/Qwen3-Coder-480B-A35B-Instruct-1M-Q4_K_M.gguf
```

This configuration creates a hard limit of 4 server slots each with ~128k context `--ctx-size 524288 -np 4`

```
llama-server \
    --model /Volumes/development2/ggufs/gpt-oss-120b-F16.gguf \
    --alias "local-llm" \
    --threads -1 \
    --seed 3407 \
    --prio 3 \
    --min_p 0.01 \
    --temp 1.0 \
    --top-p 0.95 \
    --ctx-size 524288 \
    -np 4 \
    --host  $(ipconfig getifaddr en1) \
    --port 11345 \
    -ctk q8_0 \
    -ctv q8_0 \
    --n-gpu-layers 999 \
    --split-mode layer \
    --no-mmap \
    -b 32768 \
    -ub 1024 \
    --cache-ram 0 \
    --cont-batching \
    --no-context-shift \
    --metrics \
    --log-file /tmp/local-llm.gguf.log \
    --log-timestamps \
    --jinja
```  

One server slot with 128k context


```
llama-server \
    --model /Volumes/development2/ggufs/gpt-oss-120b-F16.gguf \
    --alias "local-llm" \
    --threads -1 \
    --seed 3407 \
    --prio 3 \
    --min_p 0.01 \
    --temp 1.0 \
    --top-p 0.95 \
    --ctx-size 131072 \
    -np 1 \
    --host  $(ipconfig getifaddr en1) \
    --port 11345 \
    -ctk q8_0 \
    -ctv q8_0 \
    --n-gpu-layers 999 \
    --split-mode layer \
    --no-mmap \
    -b 32768 \
    -ub 1024 \
    --cache-ram 0 \
    --cont-batching \
    --no-context-shift \
    --metrics \
    --log-file /tmp/local-llm.gguf.log \
    --log-timestamps \
    --jinja
```


# Using Cline

https://docs.cline.bot/introduction/welcome


Cline is an open source AI coding agent that uses local AI models directly in VScode or can be used just as a cli.

```
npm install -g cline
cline auth -p openai-compatible -k llama -m local-llm -b http://llama-cpp.internal:11345/v1

cline start
```

# Using Codex

https://github.com/openai/codex

Codex CLI is a coding agent from OpenAI that runs locally on your computer, without an OpenAI account.

Create `$HOME/.codex/config.toml`

```
[model_providers.llama-local]
name = "Internal LLM via llama.cpp"
base_url = "http://llama-cpp.internal:11345/v1"
wire_api = "chat"
```

```
npm i -g @openai/codex
codex --model local-llm -c model_provider=llama-local
```

# Using claude code cli

[Here](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp) support for Anthropic Messages API `/v1/messages` endpoint was added. Which allows one to use claude code cli 100% local without an Anthrophic account. 

- https://www.anthropic.com/engineering/claude-code-best-practices


> For best results with agentic workloads, use specialized agentic coding models like Nemotron, Qwen3 Coder, Kimi K2, or MiniMax M2

```
export ANTHROPIC_AUTH_TOKEN=llama
export ANTHROPIC_BASE_URL=http://llama-cpp.internal:11345
```

`$HOME/.claude/settings.json`

```
{
  "env": {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

Run claude within an existing codebase or start new project.

```
claude --version # 2.1.12 (Claude Code)

claude --model local-llm
# run /init to generate CLAUDE.md file if does not exist.
```