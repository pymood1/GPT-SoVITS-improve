# Runtime Status Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visually rich, live hardware and training-progress dashboard to the Gradio WebUI.

**Architecture:** A dependency-light runtime-status helper owns snapshot storage, hardware sampling, and HTML rendering. Training entry points publish progress; `webui.py` renders the dashboard and refreshes it every two seconds through `gr.Timer`.

**Tech Stack:** Python, psutil already used by WebUI, PyTorch runtime, nvidia-smi best-effort query, Gradio HTML/Timer, CSS.

**Spec:** `docs/superpowers/specs/2026-11-12-runtime-status-monitor-design.md`

## Global Constraints
- Support Gradio >=4.44.1,<6 according to the existing requirements split.
- Add no dependency.
- Monitor failures must never fail training.
- Keep monitoring code separate from model logic.
- Refresh interval: 2 seconds.

## Review Focus
- CPU-only runtime must render without nvidia-smi.
- Missing/stale runtime JSON must show IDLE rather than error.
- HTML content from task labels must be escaped.
- Status updates must be atomic so Timer reads cannot observe partial JSON.
- Timer callback must use queue=False so it is not blocked behind long training jobs.

---

### Task 1: Runtime status helper

**Files:**
- Create: `tools/runtime_status.py`
- Create: `tests/test_runtime_status.py`

**Interfaces:**
- Produces: `publish_activity(...)`, `clear_activity(...)`, `read_activity(...)`, `collect_hardware_snapshot()`, `render_status_html(...)`.

- [ ] Write tests for atomic state round-trip, stale-state handling, HTML escaping and rendered progress.
- [ ] Run tests and verify RED.
- [ ] Implement the helper without importing torch at module import time.
- [ ] Run tests and verify GREEN.

### Task 2: Training progress publishers

**Files:**
- Modify: `GPT_SoVITS/s1_train.py`
- Modify: `GPT_SoVITS/s2_train.py`
- Modify: `GPT_SoVITS/s2_train_v3_lora.py`
- Modify: `tools/train_emotion_controller.py`

**Interfaces:**
- Consumes: `publish_activity` and `clear_activity` from Task 1.

- [ ] Add static regression assertions that each training path publishes progress and clears state.
- [ ] Verify tests RED.
- [ ] Publish epoch/step/progress/loss at existing training logging points; use a Lightning callback for S1.
- [ ] Verify GREEN and Python compile.

### Task 3: Gradio dashboard and responsive styling

**Files:**
- Modify: `webui.py`
- Modify: `tools/assets.py`
- Modify: `tests/test_gradio_layout_regression.py`

**Interfaces:**
- Consumes: hardware/activity snapshot and HTML renderer from Task 1.

- [ ] Assert the WebUI contains `gr.HTML`, `gr.Timer(value=2)`, and a queue-free tick callback.
- [ ] Verify test RED.
- [ ] Add the status dashboard below the project header and wire the Timer.
- [ ] Add responsive project-owned `status-*` CSS classes.
- [ ] Run UI/layout regression and compatibility CI.
