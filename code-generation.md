# Generating Code using LLM

## llama-server

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

# Using CoPilot 

```
npm install -g @github/copilot
```

```
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_PROVIDER_BASE_URL="http://192.168.4.24:8899/v1"
export COPILOT_PROVIDER_API_KEY="llama"
export COPILOT_MODEL="local-llm"
export COPILOT_OFFLINE="true"
```

```
copilot
```

Note: use [openshell](openshell) as the agent runtime. 

# Using opencode

`$HOME/.config/opencode/opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-local": {
      "name": "Llama.cpp (RTX4090)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://192.168.4.24:8899/v1"
      },
      "models": {
        "local-llm": {
          "name": "local-llm"
        }
      }
    }
  }
}
```