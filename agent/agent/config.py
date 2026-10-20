"""配置加载：从环境变量与 agent/.env 读取（不引入额外依赖）。"""

from __future__ import annotations

import os
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = AGENT_ROOT / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    """把 .env 里的键值写进 os.environ（不覆盖已有的环境变量）。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env(key: str, default: str = "") -> str:
    _load_env_file()
    return os.environ.get(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    value = get_env(key, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def get_float(key: str, default: float) -> float:
    raw = get_env(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
