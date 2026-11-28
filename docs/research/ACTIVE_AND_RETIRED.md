# Active research and retired experiments

As of 2026-11-27 KST. This index distinguishes research priority from file deletion. Historical results and code remain reproducible by pinned commit. No production model or default is changed.

## Active / keep

- v23 multi-arm screening, v21 word-aligned AR comparator, v22 NAR duration and masked model, v17 EOS weights and corpus. In-flight v23 run 46006831134 uses code fd87872f858a8e26a2da85244ef1f1cdbb4df783; request b31848880da38765e5e7c253750f006b0cb8b27c. Do not cancel or resubmit for housekeeping.
- Required Kaggle source notebooks: 45952006429 (v17), 45990209360 (v21), 45979093677 (v20), 45973236470 (v19, upstream Korean corpus), 46001023562 (v22 duration model). Retain historical output/audio and metrics, especially user listening comparisons.
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
| Scratch NAR as immediate AR replacement | Faster and better duration, poor content even with oracle length | 46001023562 |
| Initial CFM noise scale alone | .875 retained, no clear quality improvement | 46001023562 |

Retired means stop automatically repeating that direction without a new hypothesis; it does not imply that every associated source file is unreachable or that later evidence cannot reopen it. Do not delete checkpoints, audio, old notebook outputs or workflow histories merely because an arm was rejected.

## Housekeeping

v22 worker: removed unused v19 contrastive helpers, unused teacher-forcing wrapper/imports, unused frontend import, unused reference-prompt override, and unused nearest/pooled GPU feature caches. v22 launcher: removed unused v17 holdout path variables, renamed reused v21 corpus variable. The in-flight pinned experiment is unaffected; later CPU checks validate this branch cleanup. Previous experiments remain available in Git history and existing workflow profile allowlists.

Before destructive cleanup, establish whether each candidate is referenced by any launcher, shared import, regression test, or mounted Kaggle source. Prefer removing verified dead code over deleting research records. Kaggle remote output cleanup requires separate dependency review and is not performed here.

## Multi-arm interpretation

Run `python tools/analyze_s1_s2_screening.py --source s1-next-result.json --output screening.json` after v23 completes. Match by language, speaker index, target index and seed. Cluster improvements by speaker rather than treating 16 repeated utterance/seed rows as independent subjects. Report CER and identity Pareto frontier, per-seed and per-speaker directions, descriptive bootstrap intervals and unadjusted exact sign flips. Only four independent speakers per language; intervals and p-values are exploratory, not proof. Select a small set for independent heldout confirmation and blind listening. No automatic deployment.
