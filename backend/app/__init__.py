"""FastAPI 后端：把 Agent 层包装成前端可调用的 HTTP + SSE 服务。"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 backend 能 import agent 包（agent 与 backend 平级）
AGENT_ROOT = Path(__file__).resolve().parents[2] / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
