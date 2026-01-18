# DeepSeek v3.2

Downloading the DeepSeek v3.2 and creating a quantized 4-bit version.

```
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3.2
cd DeepSeek-V3.2 
pwd;du -sh
/Volumes/development4/DeepSeek-V3.2
1.3T	.
``` 

Created `convert-deepseek-v3.2_hf_to_gguf.py` from `convert_hf_to_gguf.py` and update the script adding the changes defined [here](https://www.reddit.com/r/LocalLLaMA/comments/1q1aif6/running_an_unsupported_deepseek_v32_in_llamacpp/)

Converted to gguf 16-bit (f16)

```
python3 ~/llama.cpp/convert-deepseek-v3.2_hf_to_gguf.py ./DeepSeek-V3.2 --outfile deepseek-v3.2-f16.gguf --outtype f16

ls -lh
total 2621659704
drwxr-xr-x  177 cthorn  staff   5.5K Jan  2 20:40 DeepSeek-V3.2
-rw-r--r--    1 cthorn  staff   1.2T Jan  3 13:14 deepseek-v3.2-f16.gguf

```

Create 4-bit version

```
llama-quantize deepseek-v3.2-f16.gguf deepseek-v3.2-q4_k_m.gguf Q4_K_M

ls -lh
total 3411663648
-rw-r--r--  1 cthorn  staff   1.2T Jan  3 13:14 deepseek-v3.2-f16.gguf
-rw-r--r--  1 cthorn  staff   377G Jan  3 13:39 deepseek-v3.2-q4_k_m.gguf
```
