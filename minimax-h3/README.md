# Using Minimax-H3

> MiniMax H3 is a general-purpose, omni-modal generative system. It supports unified understanding of multimodal contexts composed of text, images, video, and audio, and can generate video with native stereo audio at resolutions up to 2K and durations of up to 15 seconds.

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://huggingface.co/unsloth/MiniMax-H3-GGUF


```
DIFFUSION_MODEL="/Users/cthorn/video-llm/minimax_h3_fl2va_pruned-Q8_0.gguf"
VAE="/Users/cthorn/video-llm/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE="/Users/cthorn/video-llm/minimax_h3_audio_vae_fp32.safetensors"
LLM="/Users/cthorn/video-llm/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
```

## Generate videos

Use prompts from catalog.json generate videos using MiniMax H3

example command used to generated a video

```bash
~/stablediff/sd-cli --mode vid_gen \
  --diffusion-model minimax_h3_ref2va_pruned-Q8_0.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --prompt "$P" \
  --width 960 --height 544 --video-frames 124 --steps 25 --cfg-scale 1.0 \
  --backend te=cpu --diffusion-fa \
  --output <name-of-video>.webm
```

Had a local LLM take the command and write the generate_videos.sh script. The first round all of the videos were exactly 2sec. After figuring out how to address that, videos of various lengths were created within the generated folder.
