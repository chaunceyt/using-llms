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
    --log-file /tmp/gpt-oss-120b-F16.gguf.log \
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

Codex CLI is a coding agent from OpenAI that runs locally on your computer.

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
