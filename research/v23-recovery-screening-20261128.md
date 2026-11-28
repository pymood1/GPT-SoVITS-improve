# V23 recovery and decision record

Recovered 368/368 waveform scores (EN 208, KO 160) from Kaggle kernel `pymood1/gpt-sovits-s1-next-46006831134`, exact experiment source `fd87872f858a8e26a2da85244ef1f1cdbb4df783`. Original output-list recovery returned HTTP 429 repeatedly. Direct official KaggleHub single-file download succeeded in [run 46106678677](https://github.com/pymood1/GPT-SoVITS-improve/actions/runs/46106678677); artifact 15247619028 contains the full result and paired screening. All six workers exited 0, fixed KO token parity true, two T4s, FP32, runtime 4499.059 s, setup 249.999 s. No GPU resubmission.

## Decisions from paired screening

16 scored observations per arm (four reference speakers, two texts per speaker, two seeds). Observations are not 16 independent speakers. Metrics are exploratory.

- KO baseline CFG 1.3: CER 4.9383%, SV .809816. CFG 1.6: CER 3.7037%, SV .811171. CER improved for three of four speaker clusters but one seed worsened; SV improved both seeds; periodicity improved all four speaker clusters. Mean active HF was slightly higher, so no claim of noise removal. Retain candidate, not production default.
- Gain-normalized reference at CFG 1.3: CER 5.1440%, SV .820023. SV improved in all four speaker clusters, both seeds, 12/16 pairs. Mean active HF .0014913 vs .0017707 (15.78% lower), both seed means improved; HF is not perceptual artifact evidence. Strong follow-up candidate. Test interaction with CFG 1.6 rather than assuming additivity.
- Midpoint16: CER 4.7325%, SV .808849; comparable at screening scale, not established faster end-to-end. Euler64 CER 10.6996%; sway -1 CER 9.4650%. Both poor results concentrated in speaker 4 / second seed; reject as default. Sway -.5 and trim are secondary candidates only.
- EN AR CER 85.2770%, SV .710852. Duration band CER 83.3819%, SV .761816, both seeds improve CER and SV, no extreme duration failures. Exact predicted duration CER 83.0904%, SV .731709; all four speaker clusters improved CER, but SV worsened in two. Soft duration SV .773770. None produces usable English. Duration control is useful as guard, not solution to phonetic content.
- Greedy cap 16/16, SV .255436. Low temperature and no repetition penalty harm SV. Scratch NAR one pass SV .815676, CER 92.2741%; iterative CER 106.414%. Pretrained masked one-pass/iter/local SV .306211/.260368/.248482; reject as-is. Low confidence repair CER unchanged, SV slightly lower, generation latency increased. Do not claim masked model convergence from training CE alone.

## Next bounded experiment design

Prioritize KO (actual user-facing problem: reference resemblance, voiced artifacts) with fixed S1 tokens, fresh speakers/utterances, same noise draws, FP32, matched reference and target. Factorial CFG 1.3/1.6 × raw/level-match, optional 1.45 center and midpoint16 controls. Stratify by reference RMS; report seed/speaker cluster deltas, CER, SV, periodicity/HF proxies, total estimator calls and wall time, blind listening randomized filenames. Compare raw and level-matched evaluation fairly; preserve raw original. Verify token parity. No adoption until fresh holdout and listening.

EN: keep duration-band diagnostic guard; prioritize phonetic/semantic content with genuine parallel utterance supervision and oracle semantic/S2 controls. Do not scale failed pretrained masked adaptation solely on reduced training loss. Held-out-confidence remasking is an isolated prototype; not a validated improvement. Re-evaluate only after content model is demonstrably better than current AR baseline.

## Result durability

Direct per-file recovery plus numeric-only terminal log backup added. Existing v23 logs were not retroactively modified. Future runner changes require CPU check and GPU validation. No additional GPU job requested in this record.
