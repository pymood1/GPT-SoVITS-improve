#!/usr/bin/env bash
set -euo pipefail
python -m py_compile GPT_SoVITS/AR/models/s1_next.py tools/train_s1_next.py tools/prepare_s1_next_fleurs.py tools/train_s1_next_bridge_pilot.py GPT_SoVITS/module/s1_next_bridge.py tools/kaggle_s1_next_pilot.py tools/kaggle_s1_next_waveforms.py tools/evaluate_s1_next_waveforms.py tools/train_s1_next_prompt.py tools/kaggle_s1_next_repair.py tools/train_s1_next_ddp.py tools/kaggle_s1_next_scale.py tools/kaggle_s1_next_convergence.py tests/test_s1_next_ddp.py
python tests/test_s1_next.py -v
python tests/test_s1_next_efficiency.py -v
python tests/test_s1_next_waveforms.py -v
python tests/test_s1_next_ddp.py -v
python -m py_compile tools/audit_s1_next_transfer.py tools/kaggle_s1_next_transfer_audit.py
PYTHONPATH=.:GPT_SoVITS python tools/audit_s1_next_transfer.py --selftest
python -m py_compile tools/train_s1_progressive_ffn.py tools/kaggle_s1_progressive_ffn.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_progressive_ffn.py --selftest
python -m py_compile tools/train_s1_progressive_stack.py tools/kaggle_s1_progressive_stack.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_progressive_stack.py --selftest
python -m py_compile tools/train_s1_joint_ffn.py tools/kaggle_s1_joint_ffn.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_joint_ffn.py --selftest
python -m py_compile tools/train_s1_fullwidth_ffn.py tools/kaggle_s1_fullwidth_ffn.py tools/evaluate_s1_stack_waveforms.py tools/score_s1_stack_waveforms.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_fullwidth_ffn.py --selftest
python -m py_compile tools/audit_s1_stack_kv.py tools/kaggle_s1_stack_kv.py
PYTHONPATH=.:GPT_SoVITS python tools/audit_s1_stack_kv.py --selftest
python -m py_compile tools/train_s1_affine_rms.py tools/kaggle_s1_affine_rms.py tools/score_s1_fp32_asr.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_affine_rms.py --selftest
python -m py_compile tools/train_s1_rms_polish.py tools/evaluate_s1_rms_waveforms.py tools/kaggle_s1_rms_polish.py
PYTHONPATH=.:GPT_SoVITS python tools/train_s1_rms_polish.py --selftest
python -m py_compile tools/kaggle_s1_rms_listen.py tools/evaluate_s1_rms_listen.py
PYTHONPATH=.:GPT_SoVITS python tools/kaggle_s1_live_status.py --selftest
python -m py_compile tools/s1_waveform_frontend.py
python tools/s1_waveform_frontend.py
python -m py_compile tools/s1_inference_export.py
PYTHONPATH=.:GPT_SoVITS python tools/s1_inference_export.py
python -m py_compile tools/s1_rank_adaptive_ffn.py
PYTHONPATH=.:GPT_SoVITS python tools/s1_rank_adaptive_ffn.py
