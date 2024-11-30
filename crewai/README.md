# Using crewai with ollama



- https://docs.crewai.com/
- https://ollama.com/
- https://ollama.com/library/mistral
- https://github.com/alejandro-ao/crewai-crash-course
- https://github.com/joaomdmoura/crewAI-examples
- https://www.youtube.com/watch?v=8bAv5FQBD2M&ab_channel=codewithbrandon


## Create modelfile
 
`vi mistralModelfile`

```
FROM mistral

# Set parementers
# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
# https://aviralrma.medium.com/understanding-llm-parameters-c2db4b07f0ee
PARAMETER temperature 0.8
PARAMETER top_p 0.5

PARAMETER stop Result
PARAMETER stop Observation
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"

SYSTEM """"""

```

## Create model

```
ollama create crewai-mistral -f mistralModelfile
```

## Create .env
git clone https://github.com/alejandro-ao/crewai-crash-course
cd crewai-crash-course

`vi .env` 

```
OPENAI_API_BASE='http://localhost:11434/v1'
OPENAI_API_KEY='ollama'
EXA_API_KEY=''
OPENAI_MODEL_NAME='crewai-mistral'
```

## create python environment.

```
conda create -name crewai-crash-course python=3.11 -y
conda activate crewai-crash-course
pip install -r requirements.txt
python srv/main.py
```