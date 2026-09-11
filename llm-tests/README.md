# Testing Various LLMs

## LLMs
- Qwen3.6-27B-Q8_0 (port 8889)
- Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P (port 8899)
- Qwen3.8-Flash-Next-UD-Q4_K_XL (port 9998)
- DeepSeek-V4-Flash-0731-UD-Q3_K_XL (port 9999)

## Runtime
- OpenShell

## Prompts

### Create a Super Mario clone in JavaScript as a single HTML page. Make the game engaging and the graphics as beautiful as possible.

Results:
- mario-deepseek-v4-flash-q3_k_p.html
- mario-qwen3.8-flash-next-q4_k_xl.html
- mario-qwen3.8-27b-uncensored-q8.html

### Write a well architected http server using Go. follow all of the best practices for the language. create a folder named: `http-server-<model-alias>` and put all of the code within it.

Results:
- http-server-dv4f
- http-server-qwen3.8-fn
- http-server-qwen3.8-27b-uncensored-q8

### Design a richly crafted voxel-art environment featuring an ornate pagoda set within a vibrant garden.\nInclude diverse vegetation—especially cherry blossom trees—and ensure the composition feels lively, colorful, and visually striking.\nUse any voxel or WebGL libraries you prefer, but deliver the entire project as a single, self-contained HTML file that I can paste and open directly in Chrome.

Results:
- pagoda-deepseek-v4-flash-q3_k_p.html
- pagoda-qwen3.8-flash-next-q4_k_xl.html
- pagoda-qwen3.8-27b-uncensored-q8.html

## Namespace Watcher
Prompt: namespace-watcher.md
- namespace-watch-dv4f
- namespace-watch-qwen3.8-fn
- namespace-watch-qwen3.8-27b-uncensored