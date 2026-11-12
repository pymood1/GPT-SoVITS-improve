"""Lightweight runtime status publishing and dashboard rendering.

This module intentionally imports only the Python standard library at module
import time. Hardware-specific libraries are loaded lazily so monitoring can
never become a hard dependency of training scripts.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional


def default_status_path() -> Path:
    temp_root = Path(os.environ.get("TEMP", "TEMP"))
    return temp_root / "runtime_status.json"


def _clamp_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, numeric))


def _clamp_progress(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, numeric))


def publish_activity(
    *,
    path: Optional[Path] = None,
    task: str,
    stage: str = "running",
    progress: Optional[float] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    loss: Optional[float] = None,
    metric: Optional[str] = None,
    state: str = "running",
    extra: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Atomically publish one runtime activity snapshot.

    Returns False on any filesystem failure. Training callers should never
    treat a monitoring failure as a training failure.
    """

    target = Path(path) if path is not None else default_status_path()
    payload: dict[str, Any] = {
        "state": str(state or "running"),
        "task": str(task or "작업"),
        "stage": str(stage or "running"),
        "updated_at": time.time(),
        "pid": os.getpid(),
    }

    normalized_progress = _clamp_progress(progress)
    if normalized_progress is not None:
        payload["progress"] = normalized_progress

    for key, value in (
        ("step", step),
        ("total_steps", total_steps),
        ("epoch", epoch),
        ("total_epochs", total_epochs),
    ):
        if value is not None:
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                pass

    if loss is not None:
        try:
            payload["loss"] = float(loss)
        except (TypeError, ValueError):
            pass

    if metric:
        payload["metric"] = str(metric)

    if extra:
        for key, value in extra.items():
            if key not in payload:
                payload[str(key)] = value

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(
            f".{target.name}.tmp-{os.getpid()}"
        )
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, target)
        return True
    except OSError:
        return False


def clear_activity(
    *,
    path: Optional[Path] = None,
    reason: str = "idle",
) -> bool:
    target = Path(path) if path is not None else default_status_path()
    payload = {
        "state": "idle",
        "task": "대기 중",
        "stage": str(reason or "idle"),
        "reason": str(reason or "idle"),
        "updated_at": time.time(),
        "pid": os.getpid(),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(
            f".{target.name}.tmp-{os.getpid()}"
        )
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, target)
        return True
    except OSError:
        return False


def read_activity(
    *,
    path: Optional[Path] = None,
    stale_after: float = 45.0,
) -> dict[str, Any]:
    target = Path(path) if path is not None else default_status_path()
    if not target.is_file():
        return {
            "state": "idle",
            "task": "대기 중",
            "stage": "idle",
            "reason": "missing",
        }

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {
            "state": "idle",
            "task": "대기 중",
            "stage": "idle",
            "reason": "invalid",
        }

    if not isinstance(payload, dict):
        return {
            "state": "idle",
            "task": "대기 중",
            "stage": "idle",
            "reason": "invalid",
        }

    if payload.get("state") != "idle":
        try:
            updated_at = float(payload.get("updated_at", 0.0))
        except (TypeError, ValueError):
            updated_at = 0.0
        if stale_after > 0 and time.time() - updated_at > stale_after:
            return {
                "state": "idle",
                "task": "대기 중",
                "stage": "idle",
                "reason": "stale",
            }

    return payload


def _query_nvidia_smi() -> Optional[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    rows = []
    for raw_line in result.stdout.splitlines():
        fields = [part.strip() for part in raw_line.split(",")]
        if len(fields) < 6:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "util": float(fields[2]),
                    "used_mib": float(fields[3]),
                    "total_mib": float(fields[4]),
                    "temp": float(fields[5]),
                }
            )
        except ValueError:
            continue

    if not rows:
        return None

    used_mib = sum(row["used_mib"] for row in rows)
    total_mib = sum(row["total_mib"] for row in rows)
    average_util = sum(row["util"] for row in rows) / len(rows)
    hottest = max(row["temp"] for row in rows)
    first_name = rows[0]["name"]
    if len(rows) > 1:
        first_name = f"{first_name} +{len(rows) - 1}"

    return {
        "gpu_available": True,
        "gpu_count": len(rows),
        "gpu_name": first_name,
        "gpu_percent": _clamp_percent(average_util),
        "vram_used_gib": used_mib / 1024.0,
        "vram_total_gib": total_mib / 1024.0,
        "vram_percent": (
            _clamp_percent(100.0 * used_mib / total_mib)
            if total_mib > 0
            else None
        ),
        "gpu_temperature_c": hottest,
    }


def _query_torch_cuda() -> Optional[dict[str, Any]]:
    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None

        device_count = torch.cuda.device_count()
        used_bytes = 0
        total_bytes = 0
        names = []
        for index in range(device_count):
            with torch.cuda.device(index):
                free_bytes, device_total = torch.cuda.mem_get_info(index)
            total_bytes += int(device_total)
            used_bytes += int(device_total - free_bytes)
            names.append(torch.cuda.get_device_name(index))

        total_gib = total_bytes / (1024.0 ** 3)
        used_gib = used_bytes / (1024.0 ** 3)
        display_name = names[0] if names else "CUDA GPU"
        if len(names) > 1:
            display_name = f"{display_name} +{len(names) - 1}"

        return {
            "gpu_available": True,
            "gpu_count": device_count,
            "gpu_name": display_name,
            "gpu_percent": None,
            "vram_used_gib": used_gib,
            "vram_total_gib": total_gib,
            "vram_percent": (
                _clamp_percent(100.0 * used_gib / total_gib)
                if total_gib > 0
                else None
            ),
            "gpu_temperature_c": None,
        }
    except Exception:
        return None


def collect_hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_percent": None,
        "ram_percent": None,
        "ram_used_gib": None,
        "ram_total_gib": None,
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_name": "GPU 없음",
        "gpu_percent": None,
        "vram_used_gib": None,
        "vram_total_gib": None,
        "vram_percent": None,
        "gpu_temperature_c": None,
    }

    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot.update(
            {
                "cpu_percent": _clamp_percent(
                    psutil.cpu_percent(interval=None)
                ),
                "ram_percent": _clamp_percent(memory.percent),
                "ram_used_gib": memory.used / (1024.0 ** 3),
                "ram_total_gib": memory.total / (1024.0 ** 3),
            }
        )
    except Exception:
        pass

    gpu = _query_nvidia_smi() or _query_torch_cuda()
    if gpu:
        snapshot.update(gpu)
    return snapshot


def _format_percent(value: Any) -> str:
    numeric = _clamp_percent(value)
    if numeric is None:
        return "--"
    return f"{numeric:.0f}%"


def _format_gib(used: Any, total: Any) -> str:
    try:
        used_value = float(used)
        total_value = float(total)
    except (TypeError, ValueError):
        return "--"
    return f"{used_value:.1f} / {total_value:.1f} GiB"


def _ring_card(
    *,
    title: str,
    percent: Any,
    primary: str,
    secondary: str,
    kind: str,
) -> str:
    numeric = _clamp_percent(percent)
    ring_value = numeric if numeric is not None else 0.0
    percent_text = _format_percent(numeric)
    return (
        f'<section class="status-card status-card-{kind}">'
        f'<div class="status-ring" style="--status-value:{ring_value:.1f}">'
        f'<div class="status-ring-inner"><strong>{percent_text}</strong>'
        f'<span>{html.escape(title)}</span></div></div>'
        f'<div class="status-card-copy"><strong>{html.escape(primary)}</strong>'
        f'<span>{html.escape(secondary)}</span></div>'
        "</section>"
    )


def render_status_html(
    hardware: Mapping[str, Any],
    activity: Mapping[str, Any],
) -> str:
    gpu_available = bool(hardware.get("gpu_available"))
    gpu_name = str(hardware.get("gpu_name") or "GPU 없음")
    gpu_primary = (
        gpu_name
        if gpu_available
        else "CUDA GPU를 찾지 못했습니다"
    )
    gpu_secondary_parts = []
    if gpu_available:
        gpu_secondary_parts.append(
            "VRAM "
            + _format_gib(
                hardware.get("vram_used_gib"),
                hardware.get("vram_total_gib"),
            )
        )
        temperature = hardware.get("gpu_temperature_c")
        if temperature is not None:
            try:
                gpu_secondary_parts.append(
                    f"{float(temperature):.0f}°C"
                )
            except (TypeError, ValueError):
                pass
    gpu_secondary = " · ".join(gpu_secondary_parts) or "GPU 정보 없음"

    cpu_card = _ring_card(
        title="CPU",
        percent=hardware.get("cpu_percent"),
        primary="Processor",
        secondary="실시간 사용률",
        kind="cpu",
    )
    ram_card = _ring_card(
        title="RAM",
        percent=hardware.get("ram_percent"),
        primary=_format_gib(
            hardware.get("ram_used_gib"),
            hardware.get("ram_total_gib"),
        ),
        secondary="시스템 메모리",
        kind="ram",
    )
    gpu_card = _ring_card(
        title="GPU",
        percent=hardware.get("gpu_percent"),
        primary=gpu_primary,
        secondary=gpu_secondary,
        kind="gpu",
    )

    state = str(activity.get("state") or "idle").lower()
    running = state == "running"
    task = html.escape(str(activity.get("task") or "대기 중"))
    stage = html.escape(str(activity.get("stage") or "idle"))

    progress = _clamp_progress(activity.get("progress"))
    if progress is None:
        step = activity.get("step")
        total_steps = activity.get("total_steps")
        epoch = activity.get("epoch")
        total_epochs = activity.get("total_epochs")
        try:
            if (
                step is not None
                and total_steps is not None
                and int(total_steps) > 0
            ):
                progress = _clamp_progress(
                    int(step) / int(total_steps)
                )
            elif (
                epoch is not None
                and total_epochs is not None
                and int(total_epochs) > 0
            ):
                progress = _clamp_progress(
                    int(epoch) / int(total_epochs)
                )
        except (TypeError, ValueError, ZeroDivisionError):
            progress = None

    progress_value = progress if progress is not None else 0.0
    progress_percent = int(round(progress_value * 100.0))

    step_text = ""
    if activity.get("step") is not None:
        step_text = str(activity.get("step"))
        if activity.get("total_steps") is not None:
            step_text += f" / {activity.get('total_steps')}"
    elif activity.get("epoch") is not None:
        step_text = f"Epoch {activity.get('epoch')}"
        if activity.get("total_epochs") is not None:
            step_text += f" / {activity.get('total_epochs')}"
    else:
        step_text = stage if running else "작업 없음"

    details = []
    if activity.get("loss") is not None:
        try:
            details.append(f"loss {float(activity.get('loss')):.4f}")
        except (TypeError, ValueError):
            pass
    if activity.get("metric"):
        details.append(html.escape(str(activity.get("metric"))))
    if not details:
        details.append(
            "학습이 시작되면 진행 정보가 자동으로 표시됩니다."
            if not running
            else stage
        )

    badge = "RUNNING" if running else "IDLE"
    activity_class = "running" if running else "idle"

    activity_card = (
        f'<section class="status-activity status-activity-{activity_class}">'
        '<div class="status-activity-head">'
        f'<div><span class="status-badge"><i></i>{badge}</span>'
        f'<h3>{task}</h3></div>'
        f'<strong class="status-progress-number">{progress_percent}%</strong>'
        "</div>"
        '<div class="status-progress-track">'
        f'<div class="status-progress-fill" style="width:{progress_percent}%"></div>'
        "</div>"
        '<div class="status-activity-meta">'
        f'<strong>{html.escape(step_text)}</strong>'
        f'<span>{" · ".join(details)}</span>'
        "</div>"
        "</section>"
    )

    vram_percent = hardware.get("vram_percent")
    vram_bar = _clamp_percent(vram_percent) or 0.0
    gpu_strip = (
        '<div class="status-vram-strip">'
        '<span>VRAM</span>'
        '<div class="status-vram-track">'
        f'<div class="status-vram-fill" style="width:{vram_bar:.1f}%"></div>'
        "</div>"
        f'<strong>{_format_percent(vram_percent)}</strong>'
        "</div>"
    )

    return (
        '<div class="status-dashboard">'
        '<div class="status-dashboard-top">'
        f"{gpu_card}{cpu_card}{ram_card}"
        "</div>"
        f"{gpu_strip}"
        f"{activity_card}"
        "</div>"
    )


def build_runtime_status_html() -> str:
    try:
        hardware = collect_hardware_snapshot()
    except Exception:
        hardware = {}
    try:
        activity = read_activity()
    except Exception:
        activity = {
            "state": "idle",
            "task": "대기 중",
            "stage": "idle",
            "reason": "monitor_error",
        }
    return render_status_html(hardware, activity)
