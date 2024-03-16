# K8sGPT

- https://docs.k8sgpt.ai/getting-started/getting-started/
- https://github.com/k8sgpt-ai/k8sgpt

## Download binary

```
wget https://github.com/k8sgpt-ai/k8sgpt/releases/download/v0.3.26/k8sgpt_Darwin_x86_64.tar.gz
tar -xvzf k8sgpt_Darwin_x86_64.tar.gz
mv k8sgpt ~/bin
```

## Large Language Model (LLM)

For the Large Language Model (LLM) we're going to run it locally using Ollama and llama2:13b

Ollama: https://ollama.com/

Ollama OpenAI compatibility  https://ollama.com/blog/openai-compatibility
- https://github.com/ollama/ollama/blob/main/docs/openai.md

```
# terminal 1
ollama pull llama2:13b
export OLLAMA_DEBUG=1
export OLLAMA_HOST=<LOCAL_IPADDR>:11435
ollama serve


# terminal 2
export OLLAMA_HOST=<LOCAL_IPADDR>:11435
ollama list
k8sgpt auth update openai --model llama2:13b --baseurl http://${OLLAMA_HOST}/v1
password: ollama
```

## Operator

```
helm repo add k8sgpt https://charts.k8sgpt.ai/
helm repo update
helm upgrade -i k8sgpt k8sgpt/k8sgpt-operator -n k8sgpt-operator-system --create-namespace
kubectl create secret generic k8sgpt-sample-secret --from-literal=openai-api-key=ollama -n k8sgpt-operator-system
```