# Using llama.cpp for LLM inference

LLM inference in C/C++ (llama.cpp) is a library that allows one to efficiently run large language models (LLMs) on their standard consumer hardware, including local CPUs and GPUs. It introduced and uses the GGUF (GGML Universal Format) file format, which stores the model's weights, vocabulary, and metadata in a single self-contained file.

This `README.md` will show example usage. 

## Build llama.cpp tools

Download and [build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) [llama.cpp](https://github.com/ggml-org/llama.cpp)

Below are the compile parameters I use.
```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

conda create -m llm-work python=3.11 -y
conda activate llm-work

pip install -r requirements.txt

cmake -B build -DGGML_RPC=ON -BUILD_SHARED_LIBS=ON
cmake --build build --config Release

```

Update `$PATH`

```
export PATH=$PATH:/Users/<username>/llama.cpp/build/bin

# https://github.com/hybridgroup/yzma
export YZMA_LIB=/Users/<username>/llama.cpp/build/bin
```

### Keeping up with changes upstream

Keep up with updates, since new LLM support is added quickly, 
Monitor: https://github.com/ggml-org/llama.cpp/releases

```
git fetch --all  # use the latest tag below.
git switch -c <latest-tag> <latest-tag>

cmake -B build -DGGML_RPC=ON -BUILD_SHARED_LIBS=ON 
cmake --build build --config Release
```

## Convert LLM to gguf

Using llama.cpp tools to convert a LLM to a [gguf](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) [quantized](https://huggingface.co/docs/optimum/en/concept_guides/quantization) to f16-bit. Below will walk through downloading `gpt-oss-20b` model and converting it to gguf.

NOTE: run the `pip install -r requirements.txt` within the `$HOME/llama.cpp` before attempting to convert to gguf.

```
conda activate llm-work
git lfs install
git clone https://huggingface.co/openai/gpt-oss-20b
cd gpt-oss-20b
python3 ~/llama.cpp/convert_hf_to_gguf.py gpt-oss-20b --outfile gpt-oss-20b-f16.gguf --outtype f16
```

## Quantize the 16-bit to 8-bit and 4-bit

Using llama.cpp quantization command, we can compress a model's size and lower the memory requirements (e.g., from 16-bit to 4-bit or 8-bit precision) which allows one to run on local CPUs and GPUs.

some gguf types: "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K"

Convert a f16-bit gguf to more compressed quantizations types.
```
llama-quantize gpt-oss-20b-f16.gguf  gpt-oss-20b.q8.gguf Q8_0
llama-quantize gpt-oss-20b-f16.gguf  gpt-oss-20b.q4_K_M.gguf Q4_K_M

```
## Benchmark the LLM using llama-bench

Benchmark the gguf to determine performance of the LLM.

`benchmark-llm`

```
#!/bin/bash
MODEL=$1

echo "Text generation..."
llama-bench -ngl 999 -p 0 -n 128,256,512 -m ${MODEL}

echo "Prompt processing with different batch sizes"
llama-bench -ngl 999 -n 0 -p 1024 -b 128,256,512,1024 -m ${MODEL}

echo "Different number of threads..."
llama-bench -ngl 999 -n 0 -n 16 -p 64 -t 1,2,4,8,16,32 -m ${MODEL}
```

Running the script.

```
benchmark-llm gpt-oss-20b-f16.gguf
benchmark-llm gpt-oss-20b.q8.gguf
benchmark-llm gpt-oss-20b.q4_K_M.gguf
```

To calculate the TTFT convert the avg_ns: `<value> / 1000000 = <value>ms`.

```
llama-bench -m ggufs/gpt-oss-20b-F16.gguf -o json

...
[
  {
...
    "avg_ns": 206333800,
...
  },
  {
...
    "avg_ns": 1276850291,
...
  }
]
...

```



## Using llama-server

Example script using llama-server for inference against a single gguf. Use the `--models-dir` directive to enable the built in router server.

Connect to the endpoint using Cline, mcphost, and other tools that use an openai compatiable api.

```
#!/bin/bash

MODEL_PATH=gpt-oss-20b-f16.gguf
MODEL_ALIAS=gpt-oss:20b
PORT=$1

llama-server \
  -m $MODEL_PATH \
  --alias $MODEL_ALIAS \
  --ctx-size 131072 \
  --parallel 2 \
  --flash-attn on \
  --n-gpu-layers 999 \
  --host $(ipconfig getifaddr en1) \
  --port $PORT \
  --metrics \
  --batch-size 1024 \
  --numa isolate \
  --mlock \
  --no-mmap \
  --defrag-thold 0.2 \
  --cont-batching \
  --rope-scaling linear \
  --rope-scale 2 \
  --yarn-orig-ctx 2048 \
  --yarn-ext-factor 0.5 \
  --yarn-attn-factor 1.0 \
  --yarn-beta-slow 1.0 \
  --yarn-beta-fast 32.0 \
  --embeddings \
  --temp 0.6 \
  --min-p 0.0 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 1.0 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --verbose
```

* [Code generation](code-generation.md)

### Using mcp with the openai compatiable api

Example MCP configs for LLM that support tool calling.

AWS Knowledge MCP

```
{
    "mcpServers": {
      "aws-knowledge-mcp-server": {
            "command": "uvx",
            "args": ["fastmcp", "run", "https://knowledge-mcp.global.api.aws"]
    }
 }
}
```

Kubernetes relates

```
{
    "mcpServers": {
    "kagent-tools": {
       "command": "kagent-tools",
       "args": ["--stdio", "--kubeconfig", "/Users/<usetname>/.kube/config"]
    },
   "k8sgpt": {
      "command": "k8sgpt",
      "args": [
        "serve",
        "--mcp",
        "--port",
        "3333",
        "--backend",
        "localai"
      ],
      "env": {
        "KUBECONFIG": "/Users/<username>/.kube/config"
      }
    },
    "kubernetes": {
      "command": "npx",
      "args": [
        "-y",
        "kubernetes-mcp-server@latest"
      ]
    }
 }
}

```

Search, memory, and time MCP servers
```
{
    "mcpServers": {
      "time": {
	    "command": "docker",
	    "args": ["run", "-i", "--rm", "mcp/time"]
    },
    "duckduckgo": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "mcp/duckduckgo"
      ]
    },
    "memory": {
      "command": "docker",
      "args": ["run", "-i", "-v", "research-memory:/app/dist", "--rm", "mcp/memory"]
    }
 }
}

```

Using [mcphost](https://github.com/mark3labs/mcphost) 

> A CLI host application that enables Large Language Models (LLMs) to interact with external tools through the Model Context Protocol (MCP). Currently supports Claude, OpenAI, Google Gemini, and Ollama models.

```
mcphost --config mcp.json -m openai:gpt-oss:20b --provider-url http://<ipaddr>:<port> --provider-api-key llama
```

## Using llama-embedding

```
llama-embedding -m Qwen3-Embedding-8B-f16.gguf \
    -p "Introduce yourself. Say your exact model name, including the number, and your knowledge cutoff date." \
    --pooling last \
    --verbose-prompt
```

```
# very simple inference server config with embedding enabled.

llama-server -m Qwen3-Embedding-8B-f16.gguf \
    --embedding \
    --pooling last \
    -ub 8192
```

## Using Multi-Model LLM audio, image, and text LLM(s)

Using the `llama-mtmd-cli` command allows one to write scripts to process all of the screenshot on the desktop, etc.

Download gguf formats of LLM that perform well at OCR

```
curl -s -Lo ggufs/Qwen2.5-Omni-7B-f16.gguf "https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-f16.gguf?download=true"
curl -s -Lo ggufs/mmproj-Qwen2.5-Omni-7B-f16.gguf "https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-7B-f16.gguf?download=true"

```

Perform OCR on an image.

```
llama-mtmd-cli -m ggufs/Qwen2.5-Omni-7B-f16.gguf \
    --mmproj ggufs/mmproj-Qwen2.5-Omni-7B-f16.gguf \
    --image /Users/<username>/llmops.png \
    -p "Describe the purpose of this image. Perform an OCR before describing it."
```

```
 llama-mtmd-cli -m ggufs/Qwen3-VL-235B-A22B-Instruct-UD-Q4_K_XL-00001-of-00003.gguf \
    --mmproj ggufs/mmproj-Qwen3-VL-30B-A3B-Instruct-f16.gguf \
    --image /Users/<username>/llmops.png \
    -p "Describe the purpose of this image. Perform an OCR before describing it."
```

Serve the LLM for inference

```
llama-server --port 11434 \
    --model ggufs/Qwen3-VL-30B-A3B-Instruct-Q8_0.gguf \
    --mmproj ggufs/mmproj-Qwen3-VL-30B-A3B-Instruct-f16.gguf \
    --n-gpu-layers 99 \
    --ctx-size 81920 \
    --top-p 0.8 \
    --top-k 20 \
    --temp 0.7 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn on
```


Transcribe an audio file

```
llama-mtmd-cli -m ggufs/Qwen2.5-Omni-7B-f16.gguf \
    --mmproj ggufs/mmproj-Qwen2.5-Omni-7B-f16.gguf \
    --audio /Users/<usernamme>/blogpost.wav \
    -p "Transcribe the audio file to plan text"
```

## Using llama-diffusion-cli

Readme: https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion

Run text diffusion LLM using the `llama-diffusion-cli`

There are at least three types of text diffusion LLMs as referenced in the readme above. the following worked in my environment.

```
 llama-diffusion-cli -m ./Dream-v0-Instruct-7B.i1-Q6_K.gguf  \
    -p "write hello world server using Go and the Gin framework" \
    -ub 512 \
    --diffusion-eps 0.001 \
    --diffusion-algorithm 3 \
    --diffusion-steps 256 \
    --diffusion-visual \
    --ctx-size 32768
 ```

 ```
 llama-diffusion-cli -m ./LLaDA-8B-Instruct.f16.gguf \
    -p "write hello world server using Go and the Gin framework" \
    -ub 512 \
    --diffusion-eps 0.001 \
    --diffusion-algorithm 3 \
    --diffusion-steps 512 \
    --diffusion-visual \
    --ctx-size 32768 \
    --verbose-prompt
 ```

## Using llama-tts

Using the library tooling one can convert text to speech

Download gguf files
```
curl -Lo OuteTTS-0.2-500M-FP16.gguf "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-FP16.gguf?download=true"
curl -Lo WavTokenizer-Large-75-F16.gguf  "https://huggingface.co/ggml-org/WavTokenizer/resolve/main/WavTokenizer-Large-75-F16.gguf?download=true"
```

```
llama-tts -m OuteTTS-0.2-500M-FP16.gguf \
    --n-gpu-layers 33 \
    -mv WavTokenizer-Large-75-F16.gguf \
    -p "$(cat test-text.txt)" \
    -o my-example-audio.wav
```

## Using llama-gguf-split to perform a merge

There are times when downloading an LLM from huggingface.co and the LLM has multiple `.gguf` files that are numbered in a sequence. For example the `unsloth/DeepSeek-V3.1-Terminus-GGUF` the [Q4_K_M](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF/tree/main/Q4_K_M) contains seven .gguf files from `DeepSeek-V3.1-Terminus-Q3_K_M-00001-of-00007.gguf` to `DeepSeek-V3.1-Terminus-Q3_K_M-00007-of-00007.gguf` For these cases, there is a couple of patterns to run the LLM.

Merge the files into one file using the following steps 
* Download all of the ggufs locally
* Merge the files into one `.gguf`

```
llama-gguf-split --merge DeepSeek-V3.1-Terminus-Q3_K_M-00001-of-00007.gguf DeepSeek-V3.1-Terminus-Q3_K_M.gguf
```

Or since, the tooling supports just pointing to the first file in the sequence. e.g. `llama-server -m DeepSeek-V3.1-Terminus-Q3_K_M-00001-of-00007.gguf <other parameters>` you can inference the LLM.

## Using llama-speculative

Using speculative decoding to improve LLM inferencing. 

```
llama-speculative -m Qwen3-Coder-30B-A3B-Instruct-BF16.gguf \
    -md Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf \
    -c 262144 \
    -cd 262144 \
    -ngl 99 \
    -ngld 99 \
    --draft-max 16 \
    --draft-min 4  \
    -p "Write a Go http service with a greetings and status endpoints" \ -fa on \
    --n-predict 512 \
    --seed 42 \
    --sampling-seq k \
    --top-k 4 \
    --temp 0.6
...

encoded   11 tokens in    0.098 seconds, speed:  112.812 t/s
decoded  517 tokens in   12.144 seconds, speed:   42.571 t/s

n_draft   = 16
n_predict = 517
n_drafted = 720
n_accept  = 471
accept    = 65.417%

```
"The higher the acceptance rate, the higher the generation rate will be." ~ [here](https://www.theregister.com/2024/12/15/speculative_decoding/)

Using llama-server to serve. Use any tool the supports openai compatiable api.

```
llama-server -m Qwen3-Coder-30B-A3B-Instruct-BF16.gguf \
    -md Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf \
    -c 262144 \
    -cd 262144 \
    -ngl 99 \
    -ngld 99 \
    --draft-max 8 \
    --draft-min 4 \
    --draft-p-min 0.9 \
    --host $(ipconfig getifaddr en1)  \
    --port 8999
```

## Using llama-swap instead of built in router server

Another option for configuring multiple LLMs to be served [llama-swap](https://github.com/mostlygeek/llama-swap/releases) is a tool that will load/unload LLM based on request.

Example configuration

```
models:
  gpt-oss-20b:
    cmd: |
    llama-server --port ${PORT} -m /Volumes/development2/ggufs/gpt-oss-20b-F16.gguf
    --threads -1
    --seed 3407
    --prio 3
    --min_p 0.01
    --temp 1.0
    --top-p 0.95
    --ctx-size 524288
    -np 4
    -ctk q8_0
    -ctv q8_0
    --n-gpu-layers 99
    --split-mode layer
    --no-mmap
    -b 32768
    -ub 1024
    --cache-ram 0
    --cont-batching
    --no-context-shift
    --metrics
    --log-file /tmp/gpt-oss-20b-F16.gguf.log
    --log-timestamps
    --jinja
  gpt-oss-120b:
    cmd: |
    llama-server --port ${PORT} -m /Volumes/development2/ggufs/gpt-oss-120b-F16.gguf
    --threads -1
    --seed 3407
    --prio 3
    --min_p 0.01
    --temp 1.0
    --top-p 0.95
    --ctx-size 524288
    -np 4
    -ctk q8_0
    -ctv q8_0
    --n-gpu-layers 99
    --split-mode layer
    --no-mmap
    -b 32768
    -ub 1024
    --cache-ram 0
    --cont-batching
    --no-context-shift
    --metrics
    --log-file /tmp/gpt-oss-120b-F16.gguf.log
    --log-timestamps
    --jinja
  qwen3-coder-30b:
    cmd: |
      llama-server --port ${PORT} -m /Volumes/development2/ggufs/Qwen3-Coder-30B-A3B-Instruct-BF16.gguf
    --threads -1
    --seed 3407
    --prio 3
    --min_p 0.01
    --temp 1.0
    --top-p 0.95
    --ctx-size 524288
    -np 4
    -ctk q8_0
    -ctv q8_0
    --n-gpu-layers 99
    --split-mode layer
    --no-mmap
    -b 32768
    -ub 1024
    --cache-ram 0
    --cont-batching
    --no-context-shift
    --metrics
    --log-file /tmp/Qwen3-Coder-30B-A3B-Instruct-BF16.log
    --log-timestamps
    --jinja
  minimax-m2-q4:
    cmd: |
      llama-server --port ${PORT} -m /Volumes/development2/ggufs/MiniMax-M2-Q4_K_M.gguf
    --threads -1
    --seed 3407
    --prio 3
    --min_p 0.01
    --temp 1.0
    --top-p 0.95
    --ctx-size 131072
    -np 4
    -ctk q8_0
    -ctv q8_0
    --n-gpu-layers 99
    --split-mode layer
    --no-mmap
    -b 32768
    -ub 1024
    --cache-ram 0
    --cont-batching
    --no-context-shift
    --metrics
    --log-file /tmp/MiniMax-M2-Q4_K_M.log
    --log-timestamps
    --jinja
  glm-2-5:
    cmd: |
      llama-server --port ${PORT} -m /Volumes/development2/ggufs/GLM-4.5-Air-Q4_K_M.gguf
    --threads -1
    --seed 3407
    --prio 3
    --min_p 0.01
    --temp 1.0
    --top-p 0.95
    --ctx-size 131072
    -np 4
    -ctk q8_0
    -ctv q8_0
    --n-gpu-layers 99
    --split-mode layer
    --no-mmap
    -b 32768
    -ub 1024
    --cache-ram 0
    --cont-batching
    --no-context-shift
    --metrics
    --log-file /tmp/GLM-4.5-Air-Q4_K_M.log
    --log-timestamps
    --jinja
```


