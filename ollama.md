# Ollama

Tool that allows one to run LLMs locally

- https://ollama.com/
- https://ollama.com/library

```
ollama pull <model-name>
ollama run <model-name> --verbose
```

Environment variables

```
export OLLAMA_NUM_PARALLEL=4 # Allows each model to handle four requests at the same time
export OLLAMA_MAX_LOADED_MODELS=3 # how many models can be loaded simultaneously
```

API call example

```
curl http://localhost:11434/api/generate -d '{
"model": "<model-name>",
  "prompt":"Why is the sky blue?",
  "stream": false
}'
```


Note the tag section of each model to download the recommended. Most of the time the `-q4_K_M` quantized verion is recommend.


# Create a model from .gguf

We'll use: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

[Download](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)

## Create a Modelfile

```
FROM ./mistral-7b-instruct-v0.2.Q4_K_M.gguf

# set the tempature to 1 [higher is more creative, lower is more coherent]

PARAMETER temperature 0.2
```

## Use Ollama to create a model

```
ollama create mistral-7b-instruct-v0.2 -f <model-filename>
```

## Run the model

```
ollama run mistral-7b-instruct-v0.2 --verbose
```

# Create from huggingface repo

## Create .gguf

```
git clone git@github.com:ollama/ollama.git ollama
cd ollama/
git submodule init
git submodule update llm/llama.cpp
conda create --name ollama-llama-cpp python=3.10
conda activate ollama-llama-cpp
pip install -r llm/llama.cpp/requirements.txt
make -C llm/llama.cpp quantize
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 model
python llm/llama.cpp/convert.py ./model --outtype f16 --outfile Mistral-7B-Instruct-v0.1-converted.bin
llm/llama.cpp/quantize Mistral-7B-Instruct-v0.1-converted.bin Mistral-7B-Instruct-v0.1-q4_0.gguf  q4_0
```

## Create modelfile and run the model

```
FROM ./Mistral-7B-Instruct-v0.1-q4_0.gguf

# set the tempature to 1 [higher is more creative, lower is more coherent]

PARAMETER temperature 0.2
```

```
ollama create my-mistral-7b-instruct-v0.1 -f my-mistral-7b-instruct.modelfile

ollama run my-mistral-7b-instruct-v0.1 --verbose
```

# Running on a different port

```
OLLAMA_HOST=192.168.12.3:11434 ollama serve
```

## Use k8sgpt against local LLM

- https://github.com/k8sgpt-ai/k8sgpt
- https://github.com/k8sgpt-ai/k8sgpt/releases

```
K8SGPT_VERSION=""
wget https://github.com/k8sgpt-ai/k8sgpt/releases/download/v${K8SGPT_VERSION}/k8sgpt_Darwin_x86_64.tar.gz
k8sgpt auth # api-key = "whatever-not-needed-but-required"
k8sgpt auth update openai --model <model-name> --baseurl http://192.168.12.30:11434/v1
k8sgpt --analyze
k8sgpt --analyze --explain
```

## Using MaziyarPanahi/BioMistral-7B-GGUF

-  https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF


```
wget https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf

mv BioMistral-7B.Q4_K_M.gguf biomistral-7bq4km.gguf

```

`vi modelfile`

```
FROM ./biomistral-7bq4km.gguf

PARAMETER temperature 0.2

TEMPLATE """
<s>[INST]{{ .Prompt }}[/INST]</s>
"""
```