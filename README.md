# Using Large Language Models
Notes on using Large Language Models

My goal is to learn to fine-tune an LLM using a custom dataset on my local system.

- Ollama to run LLMs locally
- ollama llama.cpp to create .gguf versions of a specific Model on huggingface
- [llama.cpp](https://github.com/ggerganov/llama.cpp) to create .gguf versions of a specific Model on huggingface
- MLX for **fine-tuning** (was able to get the [example](https://github.com/ml-explore/mlx-examples/tree/main/lora) to work on my Mac)
- [huggingface](https://huggingface.co/) used to download [models](https://huggingface.co/models) and .gguf (i.e. https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF), and .safetensors (i.e. https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- stable-diffusion image generation from text prompt(s).
- Retrieval-Augmented Generation (RAG) "[delivers](https://www.youtube.com/watch?v=T-D1OfcDW1M) two big advantages, namely: the model gets the most up-to-date and trustworthy facts, and you can see where the model got its info, lending more credibility to what it generates."
- [Mergekit](https://github.com/arcee-ai/mergekit/tree/main) - a toolkit for merging pre-trained language models
- [crewai](https://docs.crewai.com/) - framework for creating AI agents

## Compute

Apple M3 Max chip with 16‑core CPU, 40‑core GPU, 16‑core Neural Engine with 128GB unified memory

## Using Opensource LLMs

Benefits: transparency, fine-tuning, and community
Organizations: NASA/IBM, healthcare, FinGPT
Models: LLAMA2, Mistral-7B-v0.1, Mixtral-8x7B, BioMistral-7B
Risks: Hallucinations, Bias, security

## Smaller Models
- Phi2
- TinyLLama

## Model Optionization
- quantization fp16, 
- LORA

