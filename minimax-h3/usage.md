# generate_videos.sh — Usage

Batch-generates videos from the text-to-video prompts in a catalog JSON file
(entries whose `references` array is empty), using `sd-cli` in `vid_gen` mode.

## Quick start

```bash
./generate_videos.sh                        # catalog.json, 30 steps, 24 fps
./generate_videos.sh --steps 40             # 40 diffusion steps
./generate_videos.sh --fps 24 my_catalog.json
./generate_videos.sh --steps=50 --fps=24
```

## Options

| Option        | Default       | Limit      | Meaning                                        |
| ------------- | ------------- | ---------- | ---------------------------------------------- |
| `--steps N`   | 30            | max 50     | Diffusion sample steps. Higher = better quality, slower per video. |
| `--fps N`     | 24            | —          | Output frame rate. Must be a positive integer. |
| `catalog.json`| `catalog.json`| —          | Input catalog (positional argument).           |

Values above 50 for `--steps` are clamped to 50 with a notice. Non-numeric
values for `--steps`/`--fps` are rejected.

## How video length is determined

`sd-cli` has no "seconds" parameter. Length is set entirely by two flags:

```
duration (seconds) = --video-frames / --fps
```

The original fixed command used `--video-frames 56` at 24 fps, which is only
**2.3 seconds**. This script now computes the frame count per catalog entry:

```
--video-frames = duration_seconds × fps
```

- `duration_seconds` is read from each catalog entry (defaults to 10 if the
  field is missing or invalid).
- It is **clamped to 5–15 seconds**, the target range for this catalog.
- Each run prints the computed length, e.g.
  `=== [3/14] Pocket science fair (slug: pocket-science-fair) - 9s / 216 frames ===`

### Frame counts at 24 fps

| Length     | `--video-frames` |
| ---------- | ---------------- |
| 5 s        | 120              |
| 6 s        | 144              |
| 8 s        | 192              |
| 9 s        | 216              |
| 10 s       | 240              |
| 12 s       | 288              |
| 15 s       | 360              |

## Recommendations for 5–15 second videos

1. **Set `duration_seconds` to 5–15 in the catalog.** Each entry's duration
   drives its video length; the script clamps anything outside 5–15 s. The
   current catalog entries (6–15 s) are all in range.

2. **Keep `--fps 24`.** It is `sd-cli`'s default, and MiniMax-H3 reference
   video input is also 24 fps. 24 fps is smooth enough for these prompts and
   keeps the frame count (and memory) moderate. Raising fps makes videos
   longer for the same frame count *only* if you raise frame count with it —
   prefer adjusting `duration_seconds` instead.

3. **Steps: 30 for iteration, 40–50 for final renders.** More steps improve
   detail and temporal stability at the cost of roughly linear runtime. The
   script caps at 50.

4. **Watch memory on longer videos.** Diffusion and especially VAE decode
   scale with frame count. The script already passes:
   - `--temporal-tiling` — bounds VAE decode memory for supported video VAEs
   - `--offload-to-cpu` — keeps weights in RAM to save VRAM

   If a run fails with an out-of-memory error: lower `duration_seconds`
   (e.g. 15 → 10), keep the 864×480 resolution, and reduce fps only as a
   last resort.

5. **Do not raise resolution before length is stable.** `-W 864 -H 480` is
   fixed in the script. Higher resolution multiplies memory usage on top of
   the longer frame counts; increase it only after 15 s renders succeed.

6. **Re-running overwrites outputs.** Each run rewrites `videos.md` and
   overwrites `generated/<slug>.webm` for the slugs it processes.

## Outputs

- `generated/<slug>.webm` — one video per reference-free catalog entry
  (e.g. `generated/aurora-tea-canister.webm`)
- `videos.md` — markdown table: Name | File | Time to generate. Rows are
  appended as each video finishes; failures are recorded as
  `FAILED after Ns` and the script continues with the next prompt.

## Catalog conventions

- Only entries with an empty `references` array are generated (the t2v
  prompts). Entries requiring reference images/videos/audio are skipped.
- `slug` must be unique — it becomes the output filename.
- `duration_seconds` (int, 5–15) sets that video's length.
