"""pytest 公共配置：把 agent 根目录加入 sys.path，并保证测试前数据库可用。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent.db import init_db, seed_orders  # noqa: E402
from agent.tools.gateway import set_transport  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def prepared_db():
    """建表并写入模拟订单，保证工具测试有数据可查。"""
    init_db()
    seed_orders()
    yield


@pytest.fixture(autouse=True)
def reset_tool_transport():
    """每个用例前后都把工具传输方式恢复成 direct。

    背景：`set_transport` 是模块级全局，MCP 用例改成 "mcp" 后一旦用例提前失败/被打断，
    后面的用例就会莫名其妙走 MCP（会起子进程，表现为"测试卡住等子进程"）。
    这类全局状态必须在用例边界上归零。
    """
    set_transport("direct")
    yield
    set_transport("direct")
