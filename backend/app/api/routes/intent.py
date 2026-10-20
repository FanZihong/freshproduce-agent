"""意图模型接口：读取 agent 侧真实产物（训练配置、评测报告、校准结果），不再用假数据。

数据链路：
    data/intent/models/<版本>/train_config.json  +  train_summary.json   → 版本与训练信息
    eval/reports/intent_baseline.json                                    → 三方对比指标
    eval/reports/intent_calibration.json                                 → 温度与拒答阈值
    eval/reports/intent_ood.json                                         → 域外召回与误杀率

编排逻辑放在 `agent.intent.registry`（可单测），这里只做转发。
"""

from __future__ import annotations

import sys
from datetime import datetime

from fastapi import APIRouter

from app.api.schemas import IntentTrainRequest, IntentVersionOut
from app.config import AGENT_ROOT

router = APIRouter(tags=["intent"])

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent.intent.local_model import resolve_adapter_dir  # noqa: E402
from agent.intent.registry import summary_payload  # noqa: E402


def _versions() -> list[dict]:
    active = resolve_adapter_dir()
    return summary_payload(active_dir=str(active))["versions"]


@router.get("/intent/versions", response_model=list[IntentVersionOut])
def list_versions() -> list[dict]:
    """所有可对比的版本：规则版 / few-shot 基线 / 各次 LoRA 训练（含真实指标）。"""
    return _versions()


@router.post("/intent/train", response_model=IntentVersionOut)
def train(payload: IntentTrainRequest) -> dict:
    """训练要跑几小时，不能挂在 HTTP 请求里——这里返回该执行的命令行。"""
    command = (
        "cd /mnt/d/生鲜售后agent/agent && source .venv/bin/activate && "
        "nohup python -u scripts/train_intent.py --prompt-detail catalog --max-len 0 "
        "--epochs 1 --save-steps 100 --eval-limit 0 --out-dir data/intent/models/v3 "
        "> train_v3.log 2>&1 &"
    )
    note = (payload.note or "").strip()
    if note and note != "手动触发训练":
        command = f"# {note}\n{command}"
    return {
        "id": "cli-required",
        "version": "train-on-cli",
        "status": "failed",
        "f1": 0.0,
        "latencyMs": 0,
        "costPerK": 0.0,
        "note": command,
        "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
