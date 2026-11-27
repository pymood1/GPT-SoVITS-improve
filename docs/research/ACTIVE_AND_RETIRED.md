# Active research and retired experiments

As of 2026-11-27 KST. This index distinguishes research priority from file deletion. Historical results and code remain reproducible by pinned commit. No production model or default is changed.

## Active / keep

- v22 masked semantic architecture, v21 word-aligned AR comparator, v17 EOS weights and corpus, v20/v21 bilingual holdouts. The in-flight v22 run 46001023562 uses code e3ba6695d2cbbe2ad31e5279646a6af8dfb88a46; request ae77621edb1d1d141a8d88de5b58280c536e859e. Do not cancel or resubmit for housekeeping.
- Required Kaggle source notebooks: 45952006429 (v17), 45990209360 (v21), 45979093677 (v20), 45973236470 (v19, upstream Korean corpus). Retain historical output/audio and metrics, especially user listening comparisons.
- Shared CFM solver, G2P/BERT alignment, waveform frontend, acoustic probes, ASR/SV scoring, live status, datasets, tests. Some filenames from retired experiments are still imported by current runs.

## Retired as primary research directions (retain evidence, not default models)

| Direction | Reason | Evidence |
|---|---|---|
| SSL restoration v10–v12 | Clean degradation or weak corrupted-token improvements | 45923560566, 45924697066, 45925702174 |
| Small S2 curriculum v9 | No consistently improved quality | 45921918936 |
| S1 continued CE / ranking v19 | Held-out English NLL and waveform scores worsened | 45973236470 |
| Late CFG for quality | Full CFG 1.3 better on reused KO cases | 45979093677 |
| S2 reference padding as main noise explanation | Small mixed effect | 45990209360 |
| BERT alignment as sole English fix | Exact phone parity, but CER still above 80% | 45990209360 |

Retired means stop automatically repeating that direction without a new hypothesis; it does not imply that every associated source file is unreachable or that later evidence cannot reopen it. Do not delete checkpoints, audio, old notebook outputs or workflow histories merely because an arm was rejected.

## Housekeeping

v22 worker: removed unused v19 contrastive helpers, unused teacher-forcing wrapper/imports, unused frontend import, unused reference-prompt override, and unused nearest/pooled GPU feature caches. v22 launcher: removed unused v17 holdout path variables, renamed reused v21 corpus variable. The in-flight pinned experiment is unaffected; later CPU checks validate this branch cleanup. Previous experiments remain available in Git history and existing workflow profile allowlists.

Before destructive cleanup, establish whether each candidate is referenced by any launcher, shared import, regression test, or mounted Kaggle source. Prefer removing verified dead code over deleting research records. Kaggle remote output cleanup requires separate dependency review and is not performed here.
