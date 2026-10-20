"""后端配置。"""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = APP_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent
AGENT_ROOT = PROJECT_ROOT / "agent"

API_PREFIX = "/api"

# 前端开发服务器地址
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# 流式输出：回复按小块发送，模拟逐字返回
STREAM_CHUNK_SIZE = 6
STREAM_DELAY_SECONDS = 0.02

# 工具与意图的中文名，供前端展示
TOOL_LABELS = {
    "query_order": "查询订单",
    "query_logistics": "查询物流",
    "query_policy": "查询赔付规则",
    "create_ticket": "提交工单",
}

INTENT_LABELS = {
    "query_order": "订单查询",
    "urge_logistics": "催物流",
    "bad_fruit_claim": "坏果理赔",
    "short_weight": "缺斤少两",
    "refund_consult": "退款咨询",
    "invoice": "发票开具",
    "quality_qa": "品质咨询",
    "complaint": "投诉",
    "chitchat": "闲聊",
    "human_handoff": "转人工",
    "fallback": "域外问题",
}

NODE_LABELS = {
    "rails_input": "输入护栏",
    "intent": "意图识别",
    "skill": "技能选择",
    "order": "查单分支",
    "claim": "理赔分支",
    "handoff": "转人工分支",
    "fallback": "兜底分支",
    "respond": "生成回复",
    "compress": "记忆压缩",
    "rails_execution": "执行护栏",
    "rails_output": "输出护栏",
}
