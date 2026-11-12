# Runtime Status Monitor Design

## Intent
Make the Gradio WebUI feel alive and useful during long preprocessing and training runs without adding a game or distracting controls.

## Goals
- Always show CPU, RAM, GPU utilization, VRAM usage, GPU name and temperature when available.
- Show the currently active training task with real epoch/step progress and latest loss when a training process publishes it.
- Refresh in-place every 2 seconds without queueing behind training jobs.
- Keep GPT-SoVITS training behavior unchanged; monitoring must be best-effort and never abort training.
- Work on the repository's supported Gradio 4.44.1–5.x range.

## Non-goals
- No mini-game, benchmark, process manager, or remote telemetry.
- No new external dependency.
- No parsing of rendered Gradio DOM or private Gradio CSS classes.
- No heavy charting library.

## Approach
Use one small helper, `tools/runtime_status.py`, for atomic runtime-state JSON, hardware collection, and HTML rendering. Training processes publish lightweight progress updates to `$TEMP/runtime_status.json`; the WebUI reads that state plus hardware counters and renders a card dashboard with `gr.HTML`. A `gr.Timer(value=2)` refreshes the panel with `queue=False`.

## Visual design
A responsive dashboard with:
- GPU card: utilization ring, GPU model, VRAM bar, temperature.
- CPU card: utilization ring.
- RAM card: utilization ring and used/total GiB.
- Activity card: animated RUNNING/IDLE badge, task name, epoch/step text, progress bar, loss/metric line.
- Subtle gradients, glow, pulse, and progress animations only; no large motion.

## Safety / compatibility
Monitoring failures return an unavailable/idle card instead of raising. GPU collection first uses `nvidia-smi`, then degrades gracefully. HTML values are escaped. CSS targets only project-owned `status-*` classes.
