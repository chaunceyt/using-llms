# Testing Various LLMs

## LLMs
- Qwen3.6-27B-Q8_0 (port 8889)
- Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P (port 8899)
- Qwen3.8-Flash-Next-UD-Q4_K_XL (port 9998)
- DeepSeek-V4-Flash-0731-UD-Q3_K_XL (port 9999)
- Tiel-Coder-35B-A3B-UD-Q8_K_XL
- North-Mini-Code-1.0-UD-Q8_K_XL
- granite-4.2-30b-Q8_0
- Swift-1.5-Qwen3.8-27B-Q8_0

## Runtime
- OpenShell ("required")

## Harness
- claude code (provided by openshell)

```
export ANTHROPIC_AUTH_TOKEN=llama
export ANTHROPIC_BASE_URL=http://192.168.4.24:<port>

# Using Openshell as the runtime
claude --model local-llm --dangerously-skip-permissions
```

## Generated code

All of the code generated here was by one of the stated LLMs.

### Mario clone

Prompt: Create a Super Mario clone in JavaScript as a single HTML page. Make the game engaging and the graphics as beautiful as possible.

Results:
- mario-deepseek-v4-flash-q3_k_p.html
- mario-qwen3.8-flash-next-q4_k_xl.html
- mario-qwen3.8-27b-uncensored-q8.html

### Go HTTP server 

Prompt: Write a well architected http server using Go. follow all of the best practices for the language. create a folder named: `http-server-<model-alias>` and put all of the code within it.

Results:
- [http-server-dv4f](http-server-dv4f)
- [http-server-qwen3.8-fn](http-server-qwen3.8-fn)
- [http-server-qwen3.8-27b-uncensored-q8](http-server-qwen3.8-27b-uncensored-q8)

### Voxel Pagoda Garden

Prompt: Design a richly crafted voxel-art environment featuring an ornate pagoda set within a vibrant garden.\nInclude diverse vegetation—especially cherry blossom trees—and ensure the composition feels lively, colorful, and visually striking.\nUse any voxel or WebGL libraries you prefer, but deliver the entire project as a single, self-contained HTML file that I can paste and open directly in Chrome.

Results:
- pagoda-deepseek-v4-flash-q3_k_p.html
- pagoda-qwen3.8-flash-next-q4_k_xl.html
- pagoda-qwen3.8-27b-uncensored-q8.html

### Namespace Watcher

Prompt: [namespace-watcher.md](namespace-watcher.md)

Results:
- [namespace-watch-dv4f](namespace-watch-dv4f)
- [namespace-watch-qwen3.8-fn](namespace-watch-qwen3.8-fn)
- [namespace-watch-qwen3.8-27b-uncensored](namespace-watch-qwen3.8-27b-uncensored)

### Mario-Kart game

Prompt: I need you to launch five sub-agents and help me build a triple A quality game that is a clone of Mario Kart. What I want you to do is I want you to launch these sub-agents, build the game without asking me any questions at all, and use 3JS to build the game. And once you're done, report back to me.

Results:
- [mario-kart-swift-1.5-qwen3.8-27b-q8_0](mario-kart-swift-1.5-qwen3.8-27b-q8_0)

### Skylines city builder

Prompt: [skylines-city-builder-prompt.md](skylines-city-builder-prompt.md)
Source: Extracted from https://github.com/rawprogress/fable-cities/blob/main/PROMPT.md

Results
- [skylines-swift-1.5-qwen3.8-27b-q8_0](skylines-swift-1.5-qwen3.8-27b-q8_0)