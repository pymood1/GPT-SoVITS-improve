# GPT-SoVITS-improve contributor instructions

Read `CODEX_HANDOFF.md` and `EMOTION_V5.md` before changing the runtime or emotion
pipeline. Preserve ordinary GPT-SoVITS behavior, Korean UI, complete dependency
scope, user data and checkpoints. Inspect current source and runtime evidence
before proposing fixes; keep patches focused.

## Compatibility and validation

- Support Python 3.10, 3.11, 3.12, 3.13 and 3.14. Keep version markers in
  `requirements.txt` consistent with actual imports and Colab setup.
- A syntax check does not establish import compatibility. Exercise the affected
  import path in fresh interpreters with the real version-specific dependencies.
- Separate import checks, model initialization, GPU execution and audio quality
  evidence. Missing model assets are not package-import regressions.
- Use isolated environments for dependency probes. Do not silently change the
  user's PyTorch/CUDA stack or remove dependencies to obtain a passing check.
- GPU work must run on authorized cloud resources. Low-cost CPU checks, including
  imports, syntax checks and small focused tests, may run locally in isolated uv
  environments. Do not read `.env` or copy credentials into artifacts.
- Do not transfer personal audio, transcripts or training lists for compatibility
  checks. Public source and generated fixtures suffice.

## Usage-aware work policy (user update: 2026-11-11)

- Before substantive work, read the available Codex account usage limits with the
  usage tool. Inspect both the short and weekly windows; the smaller remaining
  allowance is the practical constraint. Missing quota data means unknown, not
  unlimited. Do not read authentication files to obtain usage.
- Estimate a bounded unit of implementation, validation, review and handoff before
  starting it. Reserve enough allowance to save coherent changes and report the
  exact state. Do not start work that cannot reasonably fit that reserve.
- As a conservative default, at 5% or less remaining in either relevant window,
  do not begin another substantive unit. Save progress and explain what remains.
  At 5-15%, prefer one small, bounded unit with its validation; defer broad
  exploration, speculative experiments and large multi-agent fan-out.
- Recheck after a substantial phase or before expanding the scope when the
  allowance may be tight. Do not continuously poll usage or create monitoring
  automations. Stop expanding work when the remaining allowance is unlikely to
  cover validation and handoff; reach a safe checkpoint and record the exact
  changes, executed checks, active cloud jobs and next command.
- Prefer deterministic scripts and bounded parallel tests for repetitive version
  matrices. Use at most two children only for independent work that saves cost
  or improves correctness, choosing the lightest adequate supported model.
- Never buy credits, switch to paid resources, shut down/restart the PC, or change
  power settings because of quota. Report unfinished work honestly.

This restores task-aware usage checks only. Earlier canceled continuous usage
monitoring and quota-triggered shutdown remain canceled.
