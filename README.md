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
- Learning to use [k8sgpt-operator](https://github.com/k8sgpt-ai/k8sgpt-operator) and [k8sgpt](https://github.com/k8sgpt-ai/k8sgpt) cli
- Looking at [gptscript](https://github.com/gptscript-ai/gptscript) for AI automation.
- Working on an operator that provisions [AIChat Workspaces]](https://github.com/chaunceyt/aichat-workspace-operator) powered by Open WebUI and Ollama. An attempt to simulate a LLM as a Service.

## Compute

Apple M3 Max chip with 16‑core CPU, 40‑core GPU, 16‑core Neural Engine with 128GB unified memory

## Using Opensource LLMs

- Benefits: transparency, fine-tuning, and community
- Organizations: NASA/IBM, healthcare, FinGPT
- Risks: Hallucinations, Bias, security

## Prompting vs Fine-tuning

Methods that influence model behavior,  prompting and fine-tuning.

- [good-read](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview#prompting-vs-finetuning) from anthropic docs.

Prompting

* requires less resources
* uses the base model, which can make it cheaper
* prompts are reusable across different versions


### Example system prompts

Below are some example SYSTEM prompt that will improve response from LLM.

- good example of a SYSTEM prompt. [Claude 3.5 Sonnet](https://docs.anthropic.com/en/release-notes/system-prompts) release notes for their prompt.
- useful SYSTEM prompt [examples](https://github.com/danielmiessler/fabric/tree/main/patterns)
- Ollama's [modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) make it easy to create new models with custom SYSTEM prompts

## What can LLM do?

### Chat
Some common examples used:

- Text classification
- Sentiment analysis
- Question answering
- Language translation

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
- google/gemma2 (9b)
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