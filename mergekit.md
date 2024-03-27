# Using mergekit to merge LLMs

## Create top level directory to organize things

```
mkdir merge-llms
cd merge-llms
conda create --name merge-llms python=3.11
conda activate merge-llms
```
## Create llama.cpp quantize binary

[Quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp/
pip install -r requirements.txt
make
```
# https://www.youtube.com/watch?v=IVDNhQIzyIY&ab_channel=ArceeAI
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e . 
```

### merge-llm-config.yaml 

- https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B
- https://huggingface.co/OpenPipe/mistral-ft-optimized-1218

```
slices:
  - sources:
      - model: OpenPipe/mistral-ft-optimized-1218
        layer_range: [0, 32]
      - model: mlabonne/NeuralHermes-2.5-Mistral-7B
        layer_range: [0, 32]
merge_method: slerp
base_model: OpenPipe/mistral-ft-optimized-1218
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
```

### merge-llms.py

```
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

with open ("merge-llm-config.yaml", "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

print("Merging...")
run_merge(
        merge_config,
        out_path="./merged_models",
        options=MergeOptions(
                lora_merge_cache="/tmp",
                cuda=False,
                copy_tokenizer=True,
                lazy_unpickle=False,
                low_cpu_memory=False,
        ),
)
print("Merge complete...")
```


```
python convert.py ../../mergekit/merged_models --outtype f16 --outfile mistral-merged-v2.bin
./quantize mistral-merged-fp16.bin mistral-merged-q4_k_m.gguf q4_k_m
./quantize mistral-merged-fp16.bin mistral-merged-q5_k_m.gguf q5_k_m
```

create modelfile and create new llm `mistral-merged-q4_K_M.modelfile`

```
FROM ./mistral-merged.gguf

# Set parementers

PARAMETER temperature 0.8
PARAMETER stop Result

# Sets a custom system message to specify the behavior of
# the chat assistant

# Leaving it blank for now.

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
SYSTEM """"""
```

```
ollama create mistral-merged:v1 -f <modelfilename>
ollama run mistral-merged:v1 
```

## Using mergekit-moe

- https://towardsdatascience.com/create-mixtures-of-experts-with-mergekit-11b318c99562

```
base_model: mlabonne/AlphaMonarch-7B
experts:
  - source_model: mlabonne/AlphaMonarch-7B
    positive_prompts:
    - "chat"
    - "assistant"
    - "tell me"
    - "explain"
    - "I want"
  - source_model: beowolx/CodeNinja-1.0-OpenChat-7B
    positive_prompts:
    - "code"
    - "python"
    - "javascript"
    - "programming"
    - "algorithm"
  - source_model: SanjiWatsuki/Kunoichi-DPO-v2-7B
    positive_prompts:
    - "storywriting"
    - "write"
    - "scene"
    - "story"
    - "character"
  - source_model: mlabonne/NeuralDaredevil-7B
    positive_prompts:
    - "reason"
    - "math"
    - "mathematics"
    - "solve"
    - "count"
```


## Perform merge

```
PYTORCH_ENABLE_MPS_FALLBACK=1 mergekit-moe moe-config.yaml --copy-tokenizer merged_model/
```

## Create gguf

```
python convert.py ../../mergekit/merged_model --outfile moe.gguf --outtype q8_0
```

## Create modelfile

```
FROM ./moe.gguf

# Set parementers

PARAMETER temperature 0.5

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
```

## Create a model from a Modelfile

```
ollama create moe:v1 -f moe.modelfile
```

## Run the MoE

```
ollama run moe:v1
```
