"""当前生效的意图模型版本：release.py 写，运行时与前端读。

为什么要这份文件而不是只用环境变量：环境变量只活在一个进程里，切版本要改 .env 再重启，
而且前端 / GET /api/intent/versions 看不到"现在到底跑的是哪个版本"。
把"已放行的版本"落成一份带证据的 JSON，运行时按它解析 adapter 目录，
接口按它打 active 标记——切换、回滚、复盘三件事都变成"看一份文件"。

优先级（见 local_model.resolve_adapter_dir）：
    显式传参 > INTENT_LOCAL_DIR 环境变量 > 本文件的 adapter_dir > 默认 models/v1
环境变量仍然最高，是为了让评测、A/B、临时实验可以临时压过"已发布版本"；
线上不设环境变量时，行为由这份文件决定。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent.db import DATA_DIR

AGENT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PATH = DATA_DIR / "intent" / "active.json"


def _resolve_path(raw: str | Path) -> Path:
    """把存下来的路径解析成绝对路径（存相对路径是为了换机器、换挂载点也能用）。"""
    path = Path(raw)
    return path if path.is_absolute() else (AGENT_ROOT / path).resolve()


def _portable(path: Path | str) -> str:
    """能存相对路径就存相对路径（相对 AGENT_ROOT），否则存绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(AGENT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_active(path: Path | str | None = None) -> dict | None:
    """读"当前生效版本"；文件不存在或读不出来返回 None（调用方回落默认版本）。"""
    target = Path(path) if path else ACTIVE_PATH
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def set_active(
    version: str,
    adapter_dir: Path | str,
    *,
    evidence: dict | None = None,
    path: Path | str | None = None,
) -> dict:
    """把某个版本记为"当前生效"，并把门禁证据一起存下来（复盘时看得到凭什么放行）。"""
    target = Path(path) if path else ACTIVE_PATH
    payload = {
        "version": str(version),
        "adapter_dir": _portable(adapter_dir),
        "activated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evidence": evidence or {},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def active_adapter_dir(path: Path | str | None = None) -> Path | None:
    """当前生效版本的 adapter 目录（绝对路径）；没配置或目录不存在时返回 None。"""
    payload = load_active(path)
    raw = (payload or {}).get("adapter_dir")
    if not raw:
        return None
    resolved = _resolve_path(str(raw))
    return resolved if resolved.exists() else None


def clear_active(path: Path | str | None = None) -> bool:
    """回滚用：删掉 active.json，运行时立刻回到默认版本。"""
    target = Path(path) if path else ACTIVE_PATH
    if target.exists():
        target.unlink()
        return True
    return False
