"""CPU-only static readout diagnostic. Does not measure generated audio quality."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GPT_SoVITS'))
from AR.models.emotion_control import load_emotion_controller_checkpoint, sha256_file


def abs_stats(values):
    values = values.detach().double().abs().flatten().cpu()
    return distribution(values)


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError('Statistics require nonempty finite values')
    quantiles = torch.quantile(
        values, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64),
    )
    return {
        'mean': values.mean().item(),
        'p50': quantiles[0].item(),
        'p90': quantiles[1].item(),
        'p99': quantiles[2].item(),
        'max': values.max().item(),
    }


def correlations(vectors):
    centered = vectors.double() - vectors.double().mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1)
    matrix = []
    for i in range(len(vectors)):
        row = []
        for j in range(len(vectors)):
            if norms[i] > 0 and norms[j] > 0:
                value = centered[i] @ centered[j] / (norms[i] * norms[j])
                row.append(float(value.clamp(-1, 1)))
            else:
                row.append(None)
        matrix.append(row)
    return matrix


@torch.no_grad()
def diagnose(controller, output_weight, names):
    if controller.steering_mode != 'hidden_direction':
        raise ValueError(
            'Static diagnostic requires hidden_direction; legacy scores depend on context'
        )
    if len(names) != controller.num_emotions or len(set(names)) != len(names):
        raise ValueError('Emotion names must be unique and match controller IDs')
    weight = output_weight.detach().float().cpu()
    if weight.ndim != 2 or weight.shape[1] != controller.hidden_dim or weight.shape[0] < 2:
        raise ValueError('S1 readout must be [vocab, controller.hidden_dim]')
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite S1 output weights')
    controller = copy.deepcopy(controller).cpu().float().eval()
    ids = torch.arange(controller.num_emotions)
    hidden = torch.zeros(controller.num_emotions, controller.hidden_dim)
    raw = controller.preference_scores(hidden, weight, ids)
    varied_hidden = torch.arange(hidden.numel(), dtype=hidden.dtype).reshape_as(hidden)
    other = controller.preference_scores(varied_hidden, weight, ids)
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite preference scores')
    report = {
        'report_type': 'static_emotion_steering',
        'steering_mode': controller.steering_mode,
        'dtype': 'float32',
        'hidden_dim': controller.hidden_dim,
        'vocab_size': len(weight),
        'emotion_order': names,
        'context_independent': torch.equal(raw, other),
        'max_log_odds_shift': controller.max_log_odds_shift,
        'prefix_log_ratio_budget': controller.prefix_log_ratio_budget,
        'isotropic_random_cosine_std_reference': 1 / math.sqrt(controller.hidden_dim),
        'scope': 'All vocab rows including EOS; no sampler support or prefix budget applied. '
                 'Delta is a pre-normalization log-weight, not log(p_steered/p_base). '
                 'A constant delta on support cancels. No no-op or audio-quality verdict.',
        'emotions': [],
        'correlations': {},
    }
    for index, name in enumerate(names):
        strengths = []
        for strength in (0.5, 1.0, 1.5, 2.0):
            scaled = strength * raw[index]
            delta = scaled.clamp(
                -controller.max_log_odds_shift, controller.max_log_odds_shift,
            )
            strengths.append({
                'strength': strength,
                'clamp_saturation_fraction': float(
                    (scaled.abs() >= controller.max_log_odds_shift).double().mean()
                ),
                'delta_abs': abs_stats(delta),
                'delta_std': float(delta.double().std(unbiased=False)),
                'max_pairwise_log_odds_shift': float(delta.max() - delta.min()),
                'fraction_abs_delta_below_0_05': float(
                    (delta.abs() < 0.05).double().mean()
                ),
            })
        report['emotions'].append({
            'id': index,
            'name': name,
            'raw_abs': abs_stats(raw[index]),
            'raw_std': float(raw[index].double().std(unbiased=False)),
            'strengths': strengths,
        })
    for strength in (0.5, 1.0, 1.5, 2.0):
        report['correlations'][str(strength)] = correlations(
            (strength * raw).clamp(
                -controller.max_log_odds_shift, controller.max_log_odds_shift,
            )
        )
    return report


def summarize_trace(records, static_report):
    """Aggregate every observed sequence, never only the log's last generation."""
    if static_report.get('s1_pairing') != 'verified':
        raise ValueError('Trace summary requires a verified paired static report')
    names = static_report['emotion_order']
    if not names:
        raise ValueError('Static report has no emotion ID order')
    identities = {
        'controller_sha256': static_report['controller_sha256'],
        'bound_s1_sha256': static_report['s1_sha256'],
    }
    sequences = {}
    steps = {}
    seen = {}
    duplicates = 0
    runtime_settings = set()
    score_dtypes = set()
    exception_fail_open_steps = 0
    for record in records:
        if record.get('event') != 'emotion_steering':
            continue
        if record.get('schema_version') != 2:
            raise ValueError('Trace schema 2 required; regenerate with the current logger')
        if record.get('identity') != identities:
            raise ValueError('Trace Controller/S1 SHA-256 identity differs from static pairing')
        for field in ('steering_mode', 'max_log_odds_shift', 'prefix_log_ratio_budget'):
            if field in static_report and record.get(field) != static_report[field]:
                raise ValueError(f'Trace configuration differs from static report: {field}')
        generation = record['generation']
        step = int(record['step'])
        if step < 1 or not record.get('rows'):
            raise ValueError('Trace step must be positive with at least one row')
        key = (generation, step)
        fingerprint = hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False).encode('utf-8')
        ).digest()
        if key in seen:
            if fingerprint != seen[key]:
                raise ValueError('Conflicting trace records for the same generation/step')
            duplicates += 1
            continue
        seen[key] = fingerprint
        runtime = dict(record.get('runtime') or {})
        score_dtype = runtime.pop('score_dtype', None)
        if score_dtype is not None:
            score_dtypes.add(score_dtype)
        runtime_settings.add(json.dumps(runtime, sort_keys=True))
        exception_fail_open_steps += int(record.get('failed_open', False))
        steps.setdefault(generation, set()).add(step)
        row_ids = set()
        for row in record['rows']:
            row_id = int(row['row_id'])
            if row_id in row_ids:
                raise ValueError('Duplicate row ID within trace step')
            row_ids.add(row_id)
            emotion_id = int(row['emotion_id'])
            if not 0 <= emotion_id < len(names):
                raise ValueError('Trace emotion ID is outside the static report order')
            condition = (emotion_id, float(row['strength']))
            if not math.isfinite(condition[1]):
                raise ValueError('Trace strength must be finite')
            sequence = sequences.setdefault((generation, row_id), {
                'condition': condition, 'generation': generation, 'rows': [],
                'last_step': 0, 'final_cumulative': None,
            })
            if sequence['condition'] != condition:
                raise ValueError('Emotion or strength changed within a sampler sequence')
            for field in (
                'mean_abs_delta', 'mean_abs_log_probability_ratio',
                'base_weighted_abs_log_probability_ratio', 'total_variation',
                'cumulative_log_ratio',
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f'Nonfinite trace metric: {field}')
            if row['budget_disabled'] and not row['eligible']:
                raise ValueError('Budget-disabled row must be eligible')
            sequence['rows'].append(row)
            if step > sequence['last_step']:
                sequence['last_step'] = step
                sequence['final_cumulative'] = row['cumulative_log_ratio']
    if not sequences:
        raise ValueError('No usable emotion steering trace records')
    if len(runtime_settings) > 1 or len(score_dtypes) > 1:
        raise ValueError('Runtime dtype/device/sampling settings differ; summarize separately')

    groups = {}
    for sequence in sequences.values():
        groups.setdefault(sequence['condition'], []).append(sequence)
    summaries = []
    for (emotion_id, strength), group in sorted(groups.items()):
        rows = [row for sequence in group for row in sequence['rows']]
        eligible = sum(bool(row['eligible']) for row in rows)
        disabled = sum(bool(row['budget_disabled']) for row in rows)
        token_count = sum(int(row['non_eos_support_count']) for row in rows)
        active_rows = [
            row for row in rows if row['eligible'] and not row['budget_disabled']
        ]
        active_token_count = sum(row['non_eos_support_count'] for row in active_rows)
        summary = {
            'emotion_id': emotion_id,
            'emotion_name': names[emotion_id],
            'strength': strength,
            'sequences': len(group),
            'sampler_generations': len({item['generation'] for item in group}),
            'observed_row_steps': len(rows),
            'eligible_row_steps': eligible,
            'budget_disabled_row_steps': disabled,
            'budget_disabled_fraction_pooled': disabled / eligible if eligible else None,
            'mean_abs_delta_support_weighted': sum(
                row['mean_abs_delta'] * row['non_eos_support_count'] for row in rows
            ) / token_count if token_count else None,
            'mean_abs_delta_active_support_weighted': sum(
                row['mean_abs_delta'] * row['non_eos_support_count']
                for row in active_rows
            ) / active_token_count if active_token_count else None,
            'mean_abs_log_probability_ratio_support_weighted': sum(
                row['mean_abs_log_probability_ratio'] * row['non_eos_support_count']
                for row in rows
            ) / token_count if token_count else None,
            'base_weighted_abs_log_probability_ratio_row_step_mean': sum(
                row['base_weighted_abs_log_probability_ratio'] for row in rows
            ) / len(rows),
            'total_variation_row_step_mean': sum(
                row['total_variation'] for row in rows
            ) / len(rows),
            'total_variation_active_row_step_mean': sum(
                row['total_variation'] for row in active_rows
            ) / len(active_rows) if active_rows else None,
            'final_observed_cumulative_log_ratio': distribution([
                sequence['final_cumulative'] for sequence in group
            ]),
            'final_observed_abs_cumulative_log_ratio': abs_stats(torch.tensor([
                sequence['final_cumulative'] for sequence in group
            ], dtype=torch.float64)),
            'sequence_mean_total_variation': distribution([
                sum(row['total_variation'] for row in item['rows']) / len(item['rows'])
                for item in group
            ]),
        }
        summaries.append(summary)
    incomplete = sum(len(values) != max(values) for values in steps.values())
    warnings = []
    if incomplete:
        warnings.append(f'{incomplete} sampler generations have missing initial/intermediate steps')
    if exception_fail_open_steps:
        warnings.append(
            f'{exception_fail_open_steps} sampler steps failed open after a Controller exception'
        )
    if any(group['sequences'] < 2 for group in summaries):
        warnings.append('Some conditions have fewer than two observed sampler sequences')
    return {
        'report_type': 'emotion_steering_trace_summary',
        'controller_sha256': identities['controller_sha256'],
        's1_sha256': identities['bound_s1_sha256'],
        'sampler_generations': len(steps),
        'sequences': len(sequences),
        'runtime': {
            **json.loads(next(iter(runtime_settings))),
            'score_dtype': next(iter(score_dtypes)) if score_dtypes else None,
        },
        'exception_fail_open_steps': exception_fail_open_steps,
        'duplicate_records_ignored': duplicates,
        'incomplete_trace_generations': incomplete,
        'groups': summaries,
        'warnings': warnings,
        'scope': 'Observed sampler rows/fragments, not verified distinct sentences or '
                 'completed requests. Final means last observed per-row value. '
                 'No automatic H2/no-op or audio-quality verdict. Use several '
                 'distinct texts and fixed reference/sampling conditions.',
    }


def read_trace_logs(paths):
    records = []
    malformed = 0
    for path in paths:
        with open(path, encoding='utf-8-sig') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += int('"emotion_steering"' in line)
                    continue
                if isinstance(record, dict) and record.get('event') == 'emotion_steering':
                    records.append(record)
    return records, malformed


def write_report(report, output, inputs):
    output = Path(output)
    if output.resolve() in {Path(item).resolve() for item in inputs}:
        raise ValueError('Output must not overwrite a checkpoint, static report, or trace log')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--controller')
    mode.add_argument('--trace-log', nargs='+', help='One or more captured stderr logs')
    parser.add_argument('--s1-checkpoint')
    parser.add_argument(
        '--static-report',
        help='Verified paired static JSON for trace identity and emotion names',
    )
    parser.add_argument('--output', required=True, help='JSON file; summary is printed to stdout')
    args = parser.parse_args(argv)
    if args.trace_log:
        if not args.static_report or args.s1_checkpoint:
            parser.error('--trace-log requires --static-report and excludes --s1-checkpoint')
        static_report = json.loads(Path(args.static_report).read_text(encoding='utf-8'))
        records, malformed = read_trace_logs(args.trace_log)
        report = summarize_trace(records, static_report)
        if malformed:
            report['warnings'].append(f'{malformed} malformed/truncated trace JSON lines')
        write_report(report, args.output, [args.static_report, *args.trace_log])
        for group in report['groups']:
            print(
                f"{group['emotion_name']} strength={group['strength']}: "
                f"sequences={group['sequences']}, "
                f"budget_disabled={group['budget_disabled_fraction_pooled']}, "
                f"mean_TV={group['total_variation_row_step_mean']:.6g}"
            )
        for warning in report['warnings']:
            print(f'WARNING: {warning}')
        print(report['scope'])
        print(f'JSON: {args.output}')
        return report
    if not args.s1_checkpoint or args.static_report:
        parser.error('--controller requires --s1-checkpoint and excludes --static-report')
    controller, metadata = load_emotion_controller_checkpoint(
        args.controller, map_location='cpu',
    )
    expected = metadata['extra'].get('s1_sha256')
    if not expected:
        raise ValueError('Controller has no bound S1 SHA-256; pairing cannot be verified')
    actual = sha256_file(args.s1_checkpoint)
    if expected != actual:
        raise ValueError('Controller/S1 SHA-256 pairing mismatch')
    payload = torch.load(args.s1_checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('S1 checkpoint must contain a weight mapping')
    weights = payload.get('weight', payload.get('state_dict', payload))
    if not isinstance(weights, dict):
        raise ValueError('S1 checkpoint must contain a weight mapping')
    candidates = [
        weights[key]
        for key in ('model.ar_predict_layer.weight', 'ar_predict_layer.weight')
        if key in weights
    ]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one S1 ar_predict_layer.weight')
    names = metadata.get('emotion_names') or [str(i) for i in range(controller.num_emotions)]
    report = diagnose(controller, candidates[0], names)
    report.update({
        'controller_sha256': sha256_file(args.controller),
        's1_sha256': actual,
        's1_pairing': 'verified',
    })
    write_report(report, args.output, [args.controller, args.s1_checkpoint])
    print(f"CPU float32; {report['vocab_size']} vocab rows; pairing={report['s1_pairing']}")
    for emotion in report['emotions']:
        print(f"{emotion['name']}: |raw| {emotion['raw_abs']}")
        for row in emotion['strengths']:
            print(
                f"  strength={row['strength']}: "
                f"saturation={row['clamp_saturation_fraction']:.3%}; "
                f"|delta| {row['delta_abs']}; "
                f"fraction <0.05={row['fraction_abs_delta_below_0_05']:.3%} "
                '(exploratory threshold only)'
            )
    print('Emotion delta Pearson matrices (order: ' + ', '.join(names) + '):')
    print(json.dumps(report['correlations'], allow_nan=False))
    print(report['scope'])
    print(f'JSON: {args.output}')
    return report


if __name__ == '__main__':
    main()
