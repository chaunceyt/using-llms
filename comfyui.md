# ComfyUI

Resource: https://unsloth.ai/docs/models/qwen-image-2512

Generate video, images, 3D, audio with AI

```
conda activate llm-work
mkdir comfy_ggufs
cd comfy_ggufs

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
cd ComfyUI-GGUF
pip install -r requirements.txt
cd ../..
```

##  Download models
```
cd models

curl -L -C - -o unet/qwen-image-2512-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf
curl -L -C - -o unet/qwen-image-edit-2511-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf
 
## Text Encoder + Vision Tower + VAE   
curl -L -C - -o text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
curl -L -C - -o text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf \
  https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf
curl -L -C - -o vae/qwen_image_vae.safetensors \
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors

curl -L -C - -o ../input/sloth1.jpg \
    "https://unsloth.ai/cgi/image/_1d5a5685-2d88-44ca-b50f-ba432cd646ef_9CGCY8lvw4D9JkOdueqsk.jpeg?width=1920&quality=80&format=auto"

curl -L -C - -o ../input/sloth2.jpg \
    "https://unsloth.ai/cgi/image/UnSloth_GPU_Front_-_Confetti_ArcSk-MR4MMN215UutOFZ.png?width=1920&quality=80&format=auto"

```

## Run the UI
```
cd /Volume/development3/comfy_ggufs/ComfyUI
python main.py --listen <ip-address>
```