# Using llama.cpp

Notes from iMac

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp/
pip install -r requirements.txt
make
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 model
python3 convert.py ./model/ --outfile ./models/Mistral-7B-Instruct-v0.1-q4_0.gguf --outtype f16
./quantize  ./models/Mistral-7B-Instruct-v0.1-q4_0.gguf ./Mistral-7B-Instruct-v0.1-q4_0.gguf q4_0
# ./main -m ./Mistral-7B-Instruct-v0.1-q4_0.gguf -n 128
./main -m ./Mistral-7B-Instruct-v0.1-q4_0.gguf -n 128 -ngl 0
./server -m Mistral-7B-Instruct-v0.1-q4_0.gguf --port 8888 --host 0.0.0.0 --ctx-size 10240 --parallel 4 -ngl 0 -n 512
./server -m Mistral-7B-Instruct-v0.1-q4_0.gguf --port 8888 --host localhost --ctx-size 10240 --parallel 4 -ngl 0 -n 512
```

