#!/usr/bin/env bash
#
# generate_videos.sh - batch-generate videos from prompts in a catalog JSON file.
#
# Usage: ./generate_videos.sh [--steps N] [--fps N] [catalog.json]
#
# - Only processes entries whose "references" array is empty.
# - Saves each video to generated/<slug>.webm.
# - Records name, filename, and generation time in videos.md.
# - --steps N: diffusion steps (default 30, max 50).
# - --fps N: video frame rate (default 24).
# - Video length comes from each catalog entry's duration_seconds (clamped to
#   5-15 s); --video-frames is computed as duration_seconds * fps. See usage.md.

set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [--steps N] [--fps N] [catalog.json]"
    echo ""
    echo "  --steps N    diffusion steps (default 30, max 50)"
    echo "  --fps N      video frame rate (default 24)"
    echo "  catalog.json catalog file (default: catalog.json)"
    echo ""
    echo "Video length is taken from each entry's duration_seconds (clamped to"
    echo "5-15 s); the frame count passed to sd-cli is duration_seconds * fps."
}

JSON_FILE=""
STEPS=30
FPS=24

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            if [[ $# -lt 2 ]]; then
                echo "Error: --steps requires a value" >&2
                exit 1
            fi
            STEPS="$2"
            shift 2
            ;;
        --steps=*)
            STEPS="${1#--steps=}"
            shift
            ;;
        --fps)
            if [[ $# -lt 2 ]]; then
                echo "Error: --fps requires a value" >&2
                exit 1
            fi
            FPS="$2"
            shift 2
            ;;
        --fps=*)
            FPS="${1#--fps=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$JSON_FILE" ]]; then
                echo "Error: unexpected argument: $1" >&2
                exit 1
            fi
            JSON_FILE="$1"
            shift
            ;;
    esac
done

JSON_FILE="${JSON_FILE:-catalog.json}"

if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 1 )); then
    echo "Error: --steps must be a positive integer (got: $STEPS)" >&2
    exit 1
fi
if (( STEPS > 50 )); then
    echo "Note: --steps is capped at 50; using 50 instead of $STEPS"
    STEPS=50
fi

if ! [[ "$FPS" =~ ^[0-9]+$ ]] || (( FPS < 1 )); then
    echo "Error: --fps must be a positive integer (got: $FPS)" >&2
    exit 1
fi

OUTPUT_DIR="generated"
MD_FILE="videos.md"

SD_CLI="/Users/cthorn/stablediff/sd-cli"
DIFFUSION_MODEL="/Users/cthorn/video-llm/minimax_h3_fl2va_pruned-Q8_0.gguf"
VAE="/Users/cthorn/video-llm/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE="/Users/cthorn/video-llm/minimax_h3_audio_vae_fp32.safetensors"
LLM="/Users/cthorn/video-llm/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required (install with: brew install jq)" >&2
    exit 1
fi

if [[ ! -f "$JSON_FILE" ]]; then
    echo "Error: catalog file not found: $JSON_FILE" >&2
    exit 1
fi

if [[ ! -x "$SD_CLI" ]]; then
    echo "Error: sd-cli not found or not executable: $SD_CLI" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# One tab-separated line per entry with no references: slug, title, duration_seconds, prompt.
# @tsv escapes tabs/newlines inside fields, so one record per line is safe.
LIST_FILE="$(mktemp)"
trap 'rm -f "$LIST_FILE"' EXIT

jq -r '.[] | select((.references // []) == []) | [.slug, .title, (.duration_seconds // 10), .prompt] | @tsv' \
    "$JSON_FILE" > "$LIST_FILE"

TOTAL=$(grep -c '' "$LIST_FILE" || true)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "Error: no entries without references found in $JSON_FILE" >&2
    exit 1
fi
echo "Found $TOTAL prompt(s) without references in $JSON_FILE (steps: $STEPS, fps: $FPS)"

# Fresh videos.md with a markdown table, rows appended as each video completes.
{
    echo "# Generated Videos"
    echo ""
    echo "| Name | File | Time to generate |"
    echo "| --- | --- | --- |"
} > "$MD_FILE"

FAILED=0
N=0

while IFS=$'\t' read -r slug title duration prompt; do
    [[ -z "$slug" ]] && continue
    N=$((N + 1))
    md_title="${title//|/\\|}"
    # Length comes from the catalog, clamped to the 5-15 s target range.
    if ! [[ "$duration" =~ ^[0-9]+$ ]] || (( duration < 1 )); then
        duration=10
    fi
    if (( duration < 5 )); then
        duration=5
    elif (( duration > 15 )); then
        duration=15
    fi
    frames=$(( duration * FPS ))
    echo ""
    echo "=== [$N/$TOTAL] $title (slug: $slug) - ${duration}s / ${frames} frames ==="

    start=$(date +%s)
    if "$SD_CLI" --mode vid_gen \
        --diffusion-model "$DIFFUSION_MODEL" \
        --vae "$VAE" \
        --audio-vae "$AUDIO_VAE" \
        --llm "$LLM" \
        -p "$prompt" \
        --cfg-scale 1.0 \
        --steps "$STEPS" \
        -v \
        -W 864 -H 480 \
        --diffusion-fa \
        --offload-to-cpu \
        --rng cpu \
        --fps "$FPS" --video-frames "$frames" \
        --temporal-tiling \
        -o "$OUTPUT_DIR/$slug.webm"; then
        elapsed=$(( $(date +%s) - start ))
        echo "OK: $OUTPUT_DIR/$slug.webm (${elapsed}s)"
        printf '| %s | %s | %ss |\n' "$md_title" "$OUTPUT_DIR/$slug.webm" "$elapsed" >> "$MD_FILE"
    else
        elapsed=$(( $(date +%s) - start ))
        echo "FAILED: $slug after ${elapsed}s" >&2
        printf '| %s | %s | FAILED after %ss |\n' "$md_title" "$OUTPUT_DIR/$slug.webm" "$elapsed" >> "$MD_FILE"
        FAILED=$((FAILED + 1))
    fi
done < "$LIST_FILE"

echo ""
if [[ "$FAILED" -gt 0 ]]; then
    echo "Done with errors: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed." >&2
    exit 1
fi
echo "Done: $TOTAL/$TOTAL videos generated. Log: $MD_FILE, videos in $OUTPUT_DIR/"

