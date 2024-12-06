# Using Large Language Models
Notes on using Large Language Models

My goal is to learn to fine-tune an LLM using a custom dataset, develop RAGs, chat, and code within my local environment, without having to use GPT, Claude, and Grok.

- Ollama to run LLMs locally
- ollama llama.cpp to create .gguf versions of a specific Model on huggingface
- [llama.cpp](https://github.com/ggerganov/llama.cpp) to create .gguf versions of a specific Model on huggingface
- MLX for **fine-tuning** (was able to get the [example](https://github.com/ml-explore/mlx-examples/tree/main/lora) to work on my Mac)
- [huggingface](https://huggingface.co/) used to download [models](https://huggingface.co/models) and .gguf (i.e. https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF), and .safetensors (i.e. https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- stable-diffusion image generation from text prompt(s).
- Retrieval-Augmented Generation (RAG) "[delivers](https://www.youtube.com/watch?v=T-D1OfcDW1M) two big advantages, namely: the model gets the most up-to-date and trustworthy facts, and you can see where the model got its info, lending more credibility to what it generates."
- [Mergekit](https://github.com/arcee-ai/mergekit/tree/main) - a toolkit for merging pre-trained language models
- [crewai](https://docs.crewai.com/) - framework for creating AI agents
- Very useful frontend for Ollama is [open-webui](https://github.com/open-webui/open-webui).

## Compute

Apple M3 Max chip with 16‑core CPU, 40‑core GPU, 16‑core Neural Engine with 128GB unified memory

## Using Opensource LLMs

- Benefits: transparency, fine-tuning, and community
- Organizations: NASA/IBM, healthcare, FinGPT
- Risks: Hallucinations, Bias, security

## What can LLM do?

### Chat
Some common examples used:

- Write an email
- Summarize a book
- Plan a trip
- create a study guide
- useful [examples](https://github.com/danielmiessler/fabric/tree/main/patterns)

### Code

Many are able to perform the following functions. This will increase the number of people developing applications

- Generate
- Refactor
- Explain
- Edit
- Autocomplete

### Other
- Analyze an image
- Generate an image from text
- Recommender systems
- Generate Video
- Generate Audio

## Classes of Models

### extra small
- microsoft/Phi-3.5-mini-instruct
- meta-llama/Llama-3.2-3B-instruct
- google/gemma2:2b

### small
- google/gemmma2 (9b)
- meta-llama/Llama-3.2-8B-instruct
- qwen/qwen2..5-Coder-7B-instruct
- mistral/mistral:7b-instruct

### medium
- meta-llama/Llama-3.2-11B-Vision
- google/gemma2:27b
- llava:13b-v1.5
- qwen/qwen2.5:32b

### large
- meta-llama/Llama-3.2-70B-Instruct
- meta-llama/Llama-3.2-90B-Vision
- qwen/qwen2.5:72b

### extra large
- meta-llama/Llama-3.2-405B-Instruct
- other examples; GPT, Claude, and Grok

### models that guard

Add a layer of protection to your RAGs (input/output)

- The IBM Granite [Guardian](https://ollama.com/library/granite3-guardian) 3.0 2B and 8B models are designed to detect risks in prompts and/or responses.
- [ShieldGemma](https://ollama.com/library/shieldgemma) is set of instruction tuned models for evaluating the safety of text prompt input and text output responses against a set of defined safety policies.
- [Llama Guard](https://ollama.com/library/llama-guard3) 3 is a series of models fine-tuned for content safety classification of LLM inputs and responses
- A state-of-the-art [fact-checking](https://ollama.com/library/bespoke-minicheck) model developed by Bespoke Labs.

## Model Optionization
- quantization fp16, 
- LORA

## Some key metrics

There are two key metrics that impact user experience when interacting with LLMs. 

- latency is considered to be "time to first token" (TTFT) - constrained by how fast the model can process the input (i.e. prompt) measured in token-per-second (tok/sec)
- throughput is considered to be "time per output token" (TPOT) - can me measured by inter-token latency. (represented in tok/sec)