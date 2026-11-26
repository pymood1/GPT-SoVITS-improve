# S1 Next — independent architecture research (2026-11-26)

This branch is based on d08d895199db1fe8f5894f0775c1a99f103319ac.
The existing V5dev deployment, PR #48 and current Kaggle request are not modified.
No automatic GPU dispatch, no new public audio uploads, no production integration.

## Hypothesis and implementation
S1 Next uses pre-RMSNorm, RoPE, SDPA MHA, SwiGLU and final RMSNorm.
The 512-wide, 24-block, 16-head example is derived from the repository training
configuration; actual dimensions are read from the trusted teacher checkpoint.
For width 512, SwiGLU uses 1,344 intermediate channels (nearest multiple of
64 to 8/3 * 512), versus 2,048 for the two-matrix ReLU MLP.
This controls the dense FFN weight budget. MHA is retained to isolate quality.
Existing codebook IDs, EOS, phoneme IDs and BERT dimensions are unchanged.
Full text-prefix attention and causal audio attention are essential.

The model copies shape-compatible embeddings, BERT projection, readout and
Q/K/V/O matrices. This is NOT function-preserving: changed normalization,
positions and FFNs require distillation. Use random-init controls.
Initial implementation runs unpadded examples, FP32, single-device and AdamW.
KV-cache parity is tested. EOS is never forwarded to the acoustic codebook.

## Distillation
Use a trusted local s1v3.ckpt and already-preprocessed, separately partitioned
corpora compatible with that teacher's tokenizer, BERT and codebook.
Each folder needs 2-name2text.txt, 6-name2semantic.tsv and 3-bert/.
Do not use the personal 30-minute corpus to claim universal pretraining.
Prefer speaker-disjoint and recording-disjoint splits, with audio hashes;
the current trainer enforces distinct IDs and directories only.
Human-curated provenance/permission and split manifests are still required.
Teacher is frozen. Teacher logits and CE targets use exactly the same prefix
and prior ground-truth audio tokens. EOS position is included.
L = (1-alpha) CE + alpha * T^2 KL(teacher || student).
Initial alpha=.5, T=2, AdamW 1e-4, gradient clip 1.
These are starting points, not established optima.

Example (only after runtime/asset preparation, Kaggle or local):
```bash
python tools/train_s1_next.py --teacher /path/to/s1v3.ckpt \
  --train /path/to/train-preprocessed --validation /path/to/val-preprocessed \
  --output /kaggle/working/s1-next-research --max-steps 1000
```
Use --alpha 0 for CE-only control and --random-init for transfer ablation.
The saved best checkpoint is an experimental format, selected by held-out
teacher-forced CE; it is NOT a drop-in production .ckpt.
No resume, DDP, rollout KD, or full S2 integration yet.

## Experiment gates
1. CPU structural tests: prefix mask, causality, cache parity, EOS exclusion,
   post-token hidden alignment, gradient isolation, partial transfer, bridge.
2. Real frozen teacher load; verify checkpoint dimension and output alignment;
   real single-GPU train/eval pilot, log initial and final CE/KL, memory and time.
3. Equal token-budget ablation: old S1 fine-tuning control, Next CE-only,
   Next KD, Next random-init KD. Compare multiple seeds, speaker-disjoint data.
4. Full free-running synthesis with unchanged V5dev: CER, speaker similarity,
   duration/repetition/EOS failures, RTF, listening (blinded if possible).
   Teacher-forced CE alone is not sufficient; include prompts and long text.
5. Only after Next matches baseline, study student-generated prefix
   distillation and intermediate layer supervision. Neither is implemented.

## S1-S2 bridge
The optional SemanticResidualBridge module starts as exact hard-embedding
identity (zero scalar gate). It consumes post-token S1 hidden state, not
the predictor state for the next token. At 25 Hz, project 512 -> 128 -> 768,
add to decoded VQ embeddings before upsampling, preserving old token path.
No bridge integration or training is implemented yet. Test the bridge
separately with a fixed S1, frozen S2, target acoustic supervision via CFM loss,
then compare constrained S2 adaptation, adapter regularization and gate
strength. Use held-out real audio; compare hard-only baseline, bridge-zero,
trained bridge, and shuffled-hidden controls. Distillation alone cannot
guarantee the student knows more acoustic detail than the teacher.
