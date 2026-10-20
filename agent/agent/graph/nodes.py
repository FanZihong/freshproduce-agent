"""意图节点与三个专家分支节点。

节点约定：入参是 AgentState，返回"状态增量"字典，由 LangGraph 合并。
今天先支持单节点直接调用验证；明天在 workflow.py 里组装成图并挂条件边。
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.graph.state import AgentState, first_missing_slot
from agent.intent.decision import DecisionConfig, config_from_calibration, decide
from agent.intent.multi_intent import merge_replies
from agent.intent.multi_intent import plan as plan_multi_intent
from agent.intent.ood import collect_signals, is_out_of_domain
from agent.intent.classifier import classify
from agent.intent.taxonomy import load_taxonomy
from agent.intent.slot_extractor import extract_order_id, extract_value
from agent.memory.long_term import load_summary, save_summary
from agent.memory.short_term import compress_messages
from agent.rag.rewrite import context_from_state
from agent.tools.degradation import ASK_AGAIN
from agent.tools.gateway import ToolResult, call_tool

SLOT_QUESTIONS: dict[str, str] = {
    "order_id": "请提供订单号，我来核对订单信息。",
    "damage_ratio": "请告诉我坏果的大致比例，比如“一半”或“30%”。",
    "evidence": "请提供坏果照片；如果不方便上传，用文字描述也可以。",
}

# 需要执行的分支名 -> 节点函数（multi_node 用）
MULTI_BRANCH_NODES: dict[str, object] = {}

# 决策层配置：默认读评测产出的校准结果（intent_calibration.json），读不到就用先验默认值
_DECISION_CONFIG: DecisionConfig | None = None


def _decision_config() -> DecisionConfig:
    global _DECISION_CONFIG
    if _DECISION_CONFIG is None:
        from agent.config import get_bool
        from agent.intent.decision import load_calibration
        from agent.intent.classifier import get_classifier

        payload = load_calibration()
        config = DecisionConfig.from_dict(payload.get("config") or {})
        # 阈值是在**本地模型**的置信度分布上拟合的（intent_calibration.json 记录了 target）。
        # 拿它去判规则版 / few-shot 的置信度是口径错误——实测规则版的 0.6 置信度会被
        # 温度缩放压到 0.54，低于放行阈值，于是"澄清"会大面积误触发。所以只在
        # 校准目标与当前实际分类器一致时才启用决策层，否则一律放行（行为回到加功能之前）。
        target = payload.get("backend") or "local-lora"
        actual = get_classifier().name
        if payload.get("source") == "default" or target != actual:
            config = DecisionConfig(
                temperature=config.temperature, accept_threshold=0.0, handoff_threshold=0.0
            )

        _DECISION_CONFIG = config
        if get_bool("INTENT_DECISION_ENABLED", True) is False:
            # 关闭决策层：阈值放到 0，等价于"一律放行"，链路行为回到加这个功能之前
            _DECISION_CONFIG = DecisionConfig(
                temperature=_DECISION_CONFIG.temperature,
                accept_threshold=0.0,
                handoff_threshold=0.0,
            )
    return _DECISION_CONFIG


_RULE_CLASSIFIER = None


def _rule_label(text: str) -> str:
    """规则版标签：作为域外判定的**独立信号**（模型判 chitchat 时用它复核）。"""
    global _RULE_CLASSIFIER
    if _RULE_CLASSIFIER is None:
        from agent.intent.classifier import RuleBasedIntentClassifier

        _RULE_CLASSIFIER = RuleBasedIntentClassifier()
    labels = _RULE_CLASSIFIER.classify(text).all_labels
    return labels[0] if labels else ""


def _reset_decision_cache() -> None:
    """测试用：清掉配置缓存，让环境变量的改动生效。"""
    global _DECISION_CONFIG
    _DECISION_CONFIG = None


def intent_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """意图识别：输出 label + confidence，记录争议原因，并写入用户消息。
    新会话（还没有 summary）时尝试从长期记忆读回历史摘要。

    `config` 是 LangGraph 传进来的 RunnableConfig：会话 ID 优先取状态里的，
    没有就取 checkpointer 的 thread_id。CLI / 后端只传 `user_input`，
    不在这里兜底的话，工单与长期摘要会全部记到 "unknown" 名下（串用户数据）。

    注意注解必须是 `RunnableConfig`：写成 `dict` 时 LangGraph 会告警并且**不传 config**，
    thread_id 就拿不到，工单的 session_id 会是空串（schema 校验失败、整条链路降级）。
    """
    text = state.get("user_input", "")
    session_id = str(state.get("session_id") or "").strip()
    if not session_id:
        configurable = (config or {}).get("configurable") or {}
        session_id = str(configurable.get("thread_id") or "").strip()
    from agent.config import get_bool, get_env

    ctx = context_from_state(state)
    # 长期事实召回：只在本轮刚开始（不在补槽位）时做，且低于相似度阈值的不注入
    memory_hits = _recall_memory(state, text, session_id)
    # few-shot 分类器认 dict 上下文（dataclass 会被忽略），把事实拼进去才有意义
    classify_context = {
        "last_question": ctx.last_question,
        "last_answer": ctx.last_answer,
        "product": ctx.product,
        "memory": [hit["fact"] for hit in memory_hits],
    }
    try:
        result = classify(text, classify_context)
    except Exception as exc:  # noqa: BLE001 - 模型异常不能让整轮对话崩
        # 本地模型生成跑飞、adapter 损坏、显存不足都在这里兜住：
        # 规则版精度低，但比"这轮直接 500"好；告警留给日志排查。
        print(f"[warn] 意图模型调用失败，回退规则版：{type(exc).__name__}: {exc}")
        from agent.intent.classifier import RuleBasedIntentClassifier

        result = RuleBasedIntentClassifier().classify(text)
    # 每轮开始时把轮次级别的标志位重置，保证状态字段完整、可预测
    update: dict = {
        "session_id": session_id,
        "memory_hits": memory_hits,
        # 技能是轮次级状态：不重置的话，上一轮的技能会残留在 trace 里
        "skill_name": "",
        "skill_version": "",
        "skill_reason": "",
        "skill_score": 0.0,
        "skill_tokens": 0,
        "skill_prompt": "",
        "skill_truncated": False,
        "skill_references": [],
        "intent": result.label,
        "confidence": result.confidence,
        "handoff": False,
        "handoff_reason": "",
        "out_of_domain": False,
        "ood_reason": "",
        "decision": "accept",
        "branches": [],
        "multi_branch_inputs": {},
        "rails": [],
    }
    # rails_input 节点可能已经跑过输入规则；直接复用，避免重复记录
    rails: list[dict] = list(state.get("rails") or [])
    # 输入护栏：争议词（rails 驱动）+ PII 标记；结果写进 trace
    try:
        from agent.rails.policy import dispute_hit, pii_hit

        dispute = next(
            (item for item in rails if str(item.get("rule") or "").startswith("input.dispute")),
            None,
        )
        if dispute is None:
            dispute = dispute_hit(text)
            if dispute:
                rails.append(dispute)
        if dispute:
            update["handoff_reason"] = dispute["reason"]
        if not any(item.get("rule") == "input.pii_phone" for item in rails):
            pii = pii_hit(text)
            if pii:
                rails.append(pii)
    except Exception as exc:  # noqa: BLE001 - 护栏失败不能影响意图识别
        print(f"[warn] 输入护栏失败，按无护栏继续：{type(exc).__name__}: {exc}")

    # 1) 多意图路由计划：主/辅意图按 taxonomy 的 routing_priority 重排，产出要执行的分支
    strategy = (get_env("INTENT_MULTI_STRATEGY", "priority") or "priority").strip().lower()
    intent_plan = plan_multi_intent(result.all_labels, strategy=strategy)
    update["multi_strategy"] = intent_plan.strategy
    update["intent"] = intent_plan.primary or result.label
    # intent_labels 必须和 intent 用同一套口径（都取归一化后的计划），
    # 否则状态里会出现 intent=quality_qa 但 intent_labels=("knowledge_qa",) 的自相矛盾
    update["intent_labels"] = tuple(intent_plan.labels) or tuple(result.all_labels)
    update["intent_secondary"] = intent_plan.secondary or ""
    update["branches"] = list(intent_plan.branches)
    update["intent_plan"] = intent_plan.to_dict()

    # 2) 域外判定：只在"没在补槽位"时做，避免把订单号之类的回答判成域外
    active_flow = state.get("active_flow") or ""
    if get_bool("INTENT_OOD_GUARD", True) and not active_flow:
        signals = collect_signals(
            text,
            confidence=float(result.confidence),
            model_label=intent_plan.primary,      # 用归一化后的主意图（fallback 已映射成 chitchat）
            rule_label=_rule_label(text),
        )
        ood, reason = is_out_of_domain(signals)
        update["out_of_domain"] = ood
        update["ood_reason"] = reason
        if ood:
            from agent.rails.policy import ood_hit

            hit = ood_hit(True)
            if hit:
                rails.append(hit)

    # 3) 置信度决策：低置信度先澄清，连续澄清到上限转人工
    if not active_flow:
        clarify_rounds = int(state.get("clarify_rounds") or 0)
        decision = decide(
            float(result.confidence),
            clarify_rounds=clarify_rounds,
            config=_decision_config(),
            top_label=intent_plan.primary,
            label_name=load_taxonomy().name_of(intent_plan.primary) if intent_plan.primary else "",
        )
        update["decision"] = decision.action
        rail = (decision.metadata or {}).get("rail")
        if rail:
            rails.append(rail)
        if decision.action == "accept":
            update["clarify_rounds"] = 0
        else:
            update["clarify_rounds"] = int(decision.metadata.get("clarify_rounds") or clarify_rounds + 1)
            update["reply"] = decision.reply or (
                "抱歉，我还没完全理解您的意思，方便再说明一下您想处理的事吗？"
            )

    if not state.get("summary"):
        memory = load_summary(session_id)
        if memory:
            update["summary"] = memory

    update["rails"] = rails
    update["messages"] = [HumanMessage(content=text, id=uuid4().hex)]
    return update


def _recall_memory(state: AgentState, text: str, session_id: str) -> list[dict]:
    """从长期事实记忆里召回与当前这句话相关的条目。

    `MEMORY_RECALL=off`（默认）时直接返回空——保持现状行为不变；
    打开后：补槽位轮不召回（用户只回了订单号/比例，不需要额外上下文），
    并且只保留 `vector_score >= MEMORY_RECALL_SIM`（默认 0.55）或**命中实体**的条目。

    为什么实体命中要单独放行：短查询（"2026093000123"）的向量分往往不到阈值，
    但它和事实里的订单号完全一致，是精度最高的信号；只看向量分会把 BM25 那一路废掉。
    """
    from agent.config import get_env, get_float

    mode = (get_env("MEMORY_RECALL", "off") or "off").strip().lower()
    if mode not in {"on", "auto"}:
        return []
    if not session_id or not str(text or "").strip():
        return []
    if state.get("active_flow"):
        # 正在补槽位：这一句是碎片信息，召回没有意义还费时
        return []
    top_k = max(1, int(get_env("MEMORY_RECALL_TOPK", "3") or 3))
    try:
        from agent.memory.search import search_facts

        hits = search_facts(text, session_id, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - 记忆召回是旁路，失败按"无记忆"继续
        print(f"[warn] 事实记忆召回失败（不影响对话）：{type(exc).__name__}: {exc}")
        return []
    threshold = get_float("MEMORY_RECALL_SIM", 0.55)
    kept = [
        hit
        for hit in hits
        if float(hit.get("vector_score") or 0.0) >= threshold
        or int(hit.get("entity_hits") or 0) > 0
    ]
    return [
        {
            "id": hit["id"],
            "fact": hit["fact"],
            "score": hit["score"],
            "vector_score": hit["vector_score"],
            "entity_hits": int(hit.get("entity_hits") or 0),
            "decay": hit.get("decay"),
            "age_days": hit.get("age_days"),
        }
        for hit in kept[:top_k]
    ]


CLAUSE_SPLIT = re.compile(r"[，,。；;！!？?、\n]+")


def split_clauses(text: str) -> list[str]:
    """把一句话按标点切成子句；只有一句时返回空列表。

    多意图并行的输入通常是"诉求A，另外诉求B"，两个诉求混在一句里。
    """
    parts = [part.strip() for part in CLAUSE_SPLIT.split(text or "") if part.strip()]
    return parts if len(parts) > 1 else []


def assign_clauses(state: AgentState, branches: list[str], clauses: list[str]) -> dict[str, str]:
    """给每条分支挑一句属于它自己的子句；挑不准的分支拿不到子句。

    做法：用意图分类器（和 intent_node 同一个入口）逐句打标，再把标签映射成分支。
    这样 faq 分支检索的是「车厘子怎么保存」，而不是把整句（含「另外」「一半」）丢进知识库——
    dg-25 首跑就是整句检索，锚点落在「另外」上被判无据、误转人工，把并行策略的优点抵消掉了。
    分类器报错或没对上号时**不猜**：宁可用整句，也不要张冠李戴。
    """
    route_of = load_taxonomy().route_of
    context = context_from_state(state)
    pool = list(clauses)
    selected: dict[str, str] = {}
    for branch in branches:
        for clause in pool:
            try:
                labels = classify(clause, context).all_labels
            except Exception:
                continue
            if branch in {route_of(label) for label in labels}:
                selected[branch] = clause
                pool.remove(clause)
                break
    return selected


def multi_node(state: AgentState) -> dict:
    """并行策略出口：把 `branches` 里的两条分支都执行一遍，回复按主分支在前合并。

    为什么用"同一节点内顺序执行"而不是图级并发：两个分支节点之间没有共享写入
    （各自只读写自己的槽位与工具结果），顺序执行的**结果**与并发完全一致，
    而图级 fan-out 会让状态合并、checkpoint、错误处理都复杂一大截。
    并发调度留作后续优化，本轮的验收指标是"两条诉求是否都被处理"，与并发无关。

    每条分支先用**自己的子句**喂进去（见 assign_clauses）：不切分的话，
    知识问答分支会拿着整句去检索，命中的是另一个诉求的条目。
    """
    branches = [branch for branch in (state.get("branches") or []) if branch in MULTI_BRANCH_NODES]
    if not branches:
        return fallback_node(state)
    if len(branches) == 1:
        return MULTI_BRANCH_NODES[branches[0]](state)  # type: ignore[operator]

    base_text = state.get("user_input") or ""
    clauses = split_clauses(base_text)
    assigned = assign_clauses(state, branches, clauses) if clauses else {}

    merged: dict = {}
    replies: list[str] = []
    current: dict = dict(state)
    active_flows: list[str] = []
    pending_slots: list[str] = []
    for branch in branches:
        node = MULTI_BRANCH_NODES[branch]
        update = node({**current, "user_input": assigned.get(branch) or base_text}) or {}  # type: ignore[operator]
        merged.update({key: value for key, value in update.items() if key not in {"reply", "branches"}})
        if update.get("reply"):
            replies.append(str(update["reply"]))
        if update.get("active_flow"):
            active_flows.append(str(update["active_flow"]))
        if update.get("pending_slot"):
            pending_slots.append(str(update["pending_slot"]))
        current = {**current, **update}

    if active_flows:
        # 有分支在等槽位时要保住流程状态，否则用户下一句的槽位值会被丢掉
        merged["active_flow"] = active_flows[-1]
    if pending_slots:
        merged["pending_slot"] = pending_slots[-1]
    merged["reply"] = merge_replies(replies)
    merged["branches"] = branches
    merged["multi_branch_inputs"] = {
        branch: assigned.get(branch) or base_text for branch in branches
    }
    return merged


def rails_input_node(state: AgentState) -> dict:
    """输入护栏节点：在意图识别之前跑，命中 block 就直接给拒答话术。

    争议词/转人工仍然交给 intent 后的路由处理（保持原优先级），
    这里只负责"这句话本身不能继续处理"的情况。
    """
    text = state.get("user_input", "")
    update: dict = {
        "rails": [],
        "blocked": False,
        # 记录本轮开始时的工具数量，执行护栏只检查本轮新产生的结果
        "tool_count": len(state.get("tool_results") or []),
    }
    try:
        from agent.rails.policy import input_guard

        result = input_guard(text)
    except Exception as exc:  # noqa: BLE001 - 输入护栏失败按无护栏继续
        print(f"[warn] 输入护栏失败，按无护栏继续：{type(exc).__name__}: {exc}")
        return update
    if result is None:
        return update

    update["rails"] = list(result.hits)
    if result.blocked:
        blocked = next(
            (hit for hit in result.hits if hit.get("action") == "block"), {}
        )
        update["blocked"] = True
        update["reply"] = blocked.get("message") or "这个我没法提供，请换个问题。"
        # 被拦截的这句也要进消息历史，否则下一轮上下文缺一句
        update["messages"] = [HumanMessage(content=text, id=uuid4().hex)]
    return update


def rails_execution_node(state: AgentState) -> dict:
    """执行护栏节点：检查本轮所有成功的工具结果，异常就转人工。"""
    rails = list(state.get("rails") or [])
    try:
        from agent.rails.policy import tool_hit

        offset = int(state.get("tool_count") or 0)
        for entry in (state.get("tool_results") or [])[offset:]:
            if not entry.get("ok"):
                continue
            data = entry.get("result")
            if not isinstance(data, dict):
                continue
            hit = tool_hit(str(entry.get("name") or ""), data)
            if hit:
                rails.append(hit)
                return {
                    "handoff": True,
                    "handoff_reason": hit.get("reason") or "执行护栏拦截",
                    "reply": hit.get("message") or "这一步需要人工确认，已为您转接人工客服。",
                    "rails": rails,
                }
    except Exception as exc:  # noqa: BLE001 - 执行护栏失败不影响本轮
        print(f"[warn] 执行护栏失败，按工具成功继续：{type(exc).__name__}: {exc}")
    return {}


def rails_output_node(state: AgentState) -> dict:
    """输出护栏节点：发给用户前统一过一遍（脱敏 + 承诺/泄漏改写）。"""
    reply = state.get("reply") or ""
    if not reply:
        return {}
    try:
        from agent.rails.policy import output_guard

        text, hits = output_guard(reply)
    except Exception as exc:  # noqa: BLE001 - 输出护栏失败按原文发送
        print(f"[warn] 输出护栏失败，按原文发送：{type(exc).__name__}: {exc}")
        return {}
    update: dict = {}
    if text != reply:
        update["reply"] = text
    if hits:
        update["rails"] = [*(state.get("rails") or []), *hits]
    return update


def respond_node(state: AgentState) -> dict:
    """统一响应节点：把 reply 写成 AI 消息（输出护栏在 `rails_output` 节点）。"""
    reply = (state.get("reply") or "").strip()
    if not reply:
        return {}
    return {"messages": [AIMessage(content=reply, id=uuid4().hex)]}


def compress_node(state: AgentState) -> dict:
    """压缩节点：消息超过阈值时折叠进 summary 并删除旧消息。

    顺带在这里触发一次"事实记忆"的增量写入（`MEMORY_FACTS=rule|llm` 才生效）：
    压缩点天然是"这段对话告一段落"的位置，抽取成本被自然限流，不用每轮都调模型。
    """
    update = compress_messages(state)
    _remember_facts(state, summary=update.get("summary") or (state.get("summary") or ""))
    return update


def _remember_facts(state: AgentState, *, summary: str = "") -> None:
    """把本轮对话增量写进事实记忆；任何失败都只告警，绝不影响对话。"""
    from agent.config import get_env
    from agent.memory.extractor import turns_from_messages

    mode = (get_env("MEMORY_FACTS", "off") or "off").strip().lower()
    if mode not in {"rule", "llm"}:
        return
    session_id = str(state.get("session_id") or "").strip()
    if not session_id:
        return
    turns = turns_from_messages(state.get("messages") or [])
    if not turns:
        return
    try:
        from agent.memory.writer import remember

        remember(session_id, turns[-8:], summary=summary, mode=mode)
    except Exception as exc:  # noqa: BLE001 - 记忆是旁路，不能把对话带崩
        print(f"[warn] 事实记忆写入失败（不影响对话）：{type(exc).__name__}: {exc}")


def _tool_entry(name: str, args: dict, result: ToolResult) -> dict:
    """把工具网关的返回转成可存入状态的记录。"""
    return {
        "name": name,
        "args": args,
        "ok": result.ok,
        "result": result.data if result.ok else None,
        "error": None if result.ok else {"kind": result.kind, "message": result.message},
        "action": result.action,
        "attempts": result.attempts,
        "duration_ms": result.duration_ms,
        "transport": result.transport,
    }


def _degraded_update(
    state: AgentState,
    result: ToolResult,
    tool_results: list[dict],
    flow: str,
    retry_slot: str | None = None,
) -> dict:
    """按降级动作生成状态更新：追问 或 转人工。"""
    if result.action == ASK_AGAIN:
        return {
            "tool_results": tool_results,
            "reply": result.reply,
            "pending_slot": retry_slot or state.get("pending_slot") or "",
            "active_flow": flow,
            "loop_count": int(state.get("loop_count") or 0) + 1,
        }

    reason = result.degradation.reason if result.degradation else result.kind
    return {
        "tool_results": tool_results,
        "reply": result.reply,
        "handoff": True,
        "handoff_reason": f"{result.tool} 降级：{reason}",
        "active_flow": "",
    }


def order_node(state: AgentState) -> dict:
    """查单分支：拿订单号 -> 查订单 -> 查物流 -> 直接回答。"""
    text = state.get("user_input", "")
    slots = dict(state.get("slots") or {})
    order_id = slots.get("order_id") or extract_order_id(text)

    if not order_id:
        return {
            "pending_slot": "order_id",
            "reply": "请提供订单号，我来帮您查询物流状态。",
            "active_flow": "query_order",
        }

    order_id = str(order_id)
    tool_results = list(state.get("tool_results") or [])

    order_result = call_tool("query_order", order_id=order_id)
    tool_results.append(_tool_entry("query_order", {"order_id": order_id}, order_result))
    if not order_result.ok:
        return _degraded_update(
            state, order_result, tool_results, flow="query_order", retry_slot="order_id"
        )

    logistics_result = call_tool("query_logistics", order_id=order_id)
    tool_results.append(_tool_entry("query_logistics", {"order_id": order_id}, logistics_result))
    if not logistics_result.ok:
        return _degraded_update(state, logistics_result, tool_results, flow="query_order")

    order = order_result.data
    logistics = logistics_result.data
    reply = (
        f"订单 {order_id}：{order['product']}，实付 {float(order['amount']):.2f} 元，"
        f"当前状态：{order['status']}。{logistics['message']}。"
    )
    return {
        "slots": {**slots, "order_id": order_id},
        "tool_results": tool_results,
        "reply": reply,
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
    }


def claim_node(state: AgentState) -> dict:
    """理赔分支：按 order_id -> damage_ratio -> evidence 顺序填槽，
    槽位齐全后查赔付规则、建工单并给出结论。"""
    text = state.get("user_input", "")
    slots = dict(state.get("slots") or {})
    loop_count = int(state.get("loop_count") or 0)

    missing = first_missing_slot(slots)
    if missing:
        value = extract_value(missing, text, slots)
        if value is None:
            repeated = state.get("pending_slot") == missing
            return {
                "slots": slots,
                "pending_slot": missing,
                "loop_count": loop_count + 1 if repeated else loop_count,
                "reply": SLOT_QUESTIONS[missing],
                "active_flow": "bad_fruit_claim",
            }
        slots[missing] = value

    next_missing = first_missing_slot(slots)
    if next_missing:
        return {
            "slots": slots,
            "pending_slot": next_missing,
            "reply": SLOT_QUESTIONS[next_missing],
            "active_flow": "bad_fruit_claim",
        }

    tool_results = list(state.get("tool_results") or [])

    order_result = call_tool("query_order", order_id=str(slots["order_id"]))
    tool_results.append(_tool_entry("query_order", {"order_id": slots["order_id"]}, order_result))
    if not order_result.ok:
        if order_result.action == ASK_AGAIN:
            slots["order_id"] = None
            return {
                "slots": slots,
                "tool_results": tool_results,
                "pending_slot": "order_id",
                "loop_count": loop_count + 1,
                "reply": order_result.reply,
                "active_flow": "bad_fruit_claim",
            }
        return _degraded_update(state, order_result, tool_results, flow="bad_fruit_claim")

    order = order_result.data
    ratio = float(slots["damage_ratio"])

    policy_result = call_tool(
        "query_policy",
        amount=float(order["amount"]),
        ratio=ratio,
        category=str(order["product"]),
    )
    tool_results.append(
        _tool_entry("query_policy", {"ratio": ratio, "amount": order["amount"]}, policy_result)
    )
    if not policy_result.ok:
        if policy_result.action == ASK_AGAIN:
            slots["damage_ratio"] = None
            return {
                "slots": slots,
                "tool_results": tool_results,
                "pending_slot": "damage_ratio",
                "reply": policy_result.reply,
                "active_flow": "bad_fruit_claim",
            }
        return _degraded_update(state, policy_result, tool_results, flow="bad_fruit_claim")

    policy = policy_result.data

    ticket_result = call_tool(
        "create_ticket",
        session_id=state.get("session_id", "unknown"),
        order_id=str(slots["order_id"]),
        reason="坏果理赔",
        evidence=str(slots["evidence"]),
        refund_amount=float(policy["refund_amount"]),
    )
    tool_results.append(_tool_entry("create_ticket", {"order_id": slots["order_id"]}, ticket_result))
    if not ticket_result.ok:
        if ticket_result.action == ASK_AGAIN:
            return {
                "slots": slots,
                "tool_results": tool_results,
                "reply": ticket_result.reply,
                "active_flow": "bad_fruit_claim",
            }
        return _degraded_update(state, ticket_result, tool_results, flow="bad_fruit_claim")

    ticket = ticket_result.data

    slots["refund_amount"] = f"{float(policy['refund_amount']):.2f} 元"
    reply = (
        f"已核对订单：{order['product']}，实付 {float(order['amount']):.2f} 元。"
        f"按赔付标准，坏果比例 {ratio:.0%} 对应赔付比例 {float(policy['refund_ratio']):.0%}，"
        f"应赔付 {float(policy['refund_amount']):.2f} 元。"
        f"工单号 {ticket['ticket_id']}，预计 2 小时内审核完成。"
    )
    # 没有 session_id 就不写长期记忆：写进 "unknown" 会把不同用户的摘要混在一起
    claim_session = str(state.get("session_id") or "").strip()
    if claim_session:
        save_summary(
            claim_session,
            f"{datetime.now():%Y-%m-%d %H:%M} 完成坏果理赔：订单 {slots['order_id']}，"
            f"坏果 {ratio:.0%}，赔付 {float(policy['refund_amount']):.2f} 元，"
            f"工单 {ticket['ticket_id']}",
        )
    return {
        "slots": slots,
        "tool_results": tool_results,
        "reply": reply,
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
    }


def build_handoff_summary(state: AgentState) -> str:
    """把订单、槽位、工单和诉求拼成交接摘要。"""
    slots = state.get("slots") or {}
    parts: list[str] = []

    if slots.get("order_id"):
        parts.append(f"订单 {slots['order_id']}")
    if slots.get("damage_ratio") is not None:
        parts.append(f"坏果约 {float(slots['damage_ratio']):.0%}")
    if slots.get("evidence"):
        parts.append(f"凭证：{slots['evidence']}")
    if slots.get("refund_amount"):
        parts.append(f"系统已判定赔付 {slots['refund_amount']}")

    tool_results = state.get("tool_results") or []
    tickets = [
        item for item in tool_results if item.get("name") == "create_ticket" and item.get("ok")
    ]
    if tickets:
        parts.append(f"已建工单 {(tickets[-1].get('result') or {}).get('ticket_id')}")

    failures = [item for item in tool_results if item.get("ok") is False]
    if failures:
        last_failure = failures[-1]
        kind = (last_failure.get("error") or {}).get("kind")
        parts.append(f"异常：{last_failure.get('name')}（{kind}）")

    parts.append(f"诉求：{state.get('handoff_reason') or '用户请求人工'}")
    if state.get("user_input"):
        parts.append(f"最后一句：{state['user_input']}")
    return "；".join(parts)


def handoff_node(state: AgentState) -> dict:
    """转人工分支：生成交接摘要并标记 handoff。
    如果前面节点已经写了降级话术，就沿用那句话，不再套用通用文案。"""
    reason = state.get("handoff_reason")
    if not reason and int(state.get("loop_count") or 0) >= 3:
        reason = "连续多轮未能获取有效信息"
    summary = build_handoff_summary({**state, "handoff_reason": reason})
    reply = state.get("reply") or (
        "已为您转接人工客服，本次对话的订单信息与处理记录会一并交接，您无需重复描述。"
    )
    handoff_session = str(state.get("session_id") or "").strip()
    if handoff_session:
        save_summary(
            handoff_session,
            f"{datetime.now():%Y-%m-%d %H:%M} 转人工：{summary}",
        )
    # 会话结束也是一次自然的记忆落盘点
    _remember_facts(state, summary=f"转人工：{summary}")
    return {
        "handoff": True,
        "handoff_reason": reason or "用户请求人工",
        "handoff_summary": summary,
        "reply": reply,
        "active_flow": "",
    }


def skill_node(state: AgentState) -> dict:
    """技能节点：选技能 → 按 token 预算加载正文 → 生成注入块。

    只做"选 + 装"，不做路由、也不直接回复：
    - 路由仍由 `route_after_skill` 决定（faq / fallback / handoff）；
    - FAQ 与兜底分支在需要时通过 `compose_answer` 使用 `skill_prompt`；
    - 技能是旁路，任何失败都只告警，不能把对话带崩。
    """
    try:
        from agent.skills.loader import load_skill
        from agent.skills.prompt import build_skill_prompt
        from agent.skills.references import (
            build_reference_prompt,
            load_references,
            select_references,
            wants_details,
        )
        from agent.skills.router import select_skill

        text = state.get("user_input", "")
        match = select_skill(text)
        if match is None:
            return {}
        loaded = load_skill(match.skill)
        prompt = build_skill_prompt(loaded)
        # 二层加载：只有用户追问细节时才读 references/（白名单 + 路径校验 + 大小限制）
        references: list[dict] = []
        if loaded.skill.references and wants_details(text):
            try:
                names = select_references(loaded.skill, text)
                docs = load_references(loaded.skill, names)
                references = [doc.to_dict() for doc in docs]
                block = build_reference_prompt(loaded.skill, docs)
                if block:
                    prompt = f"{prompt}\n\n{block}"
            except Exception as exc:  # noqa: BLE001 - 细则失败不影响技能正文
                print(f"[warn] 技能细则加载失败（继续用正文）：{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - 技能是旁路
        print(f"[warn] 技能加载失败（不影响对话）：{type(exc).__name__}: {exc}")
        return {}

    return {
        "skill_name": loaded.skill.name,
        "skill_version": loaded.skill.version,
        "skill_reason": match.reason,
        "skill_score": float(match.score),
        "skill_tokens": int(loaded.tokens),
        "skill_prompt": prompt,
        "skill_truncated": bool(loaded.truncated),
        "skill_references": references,
    }


def fallback_node(state: AgentState) -> dict:
    """域外兜底：礼貌拒答并说明能力边界。

    如果这一轮命中了技能（且 `SKILL_ANSWER=llm`），先用技能作答；
    否则保持原来的固定话术，保证默认行为不变。
    """
    skill_prompt = state.get("skill_prompt") or ""
    if skill_prompt:
        from agent.skills.compose import compose_answer

        composed = compose_answer(state.get("user_input", ""), skill_prompt, [])
        if composed:
            return {"reply": composed, "active_flow": ""}
    return {
        "reply": "这个问题超出了售后服务的范围。我可以帮您查询订单物流、处理坏果理赔，或者在需要时转接人工客服。",
        "active_flow": "",
    }


def faq_node(state: AgentState) -> dict:
    """知识问答分支：只引用检索到的原文作答，无据就转人工。"""
    text = state.get("user_input", "")
    ctx = context_from_state(state)
    tool_context = (
        {"last_question": ctx.last_question, "product": ctx.product}
        if (ctx.last_question or ctx.product)
        else None
    )

    result = call_tool("search_faq", query=text, context=tool_context, top_k=3)
    tool_results = [
        *(state.get("tool_results") or []),
        _tool_entry("search_faq", {"query": text, "top_k": 3}, result),
    ]
    if not result.ok:
        return _degraded_update(state, result, tool_results, flow="knowledge_qa")

    data = result.data
    # 检索护栏：是否"无据转人工"由 rail 决定（规则缺失时自动回退 found 判断）
    try:
        from agent.rails.policy import retrieval_hit

        retrieval = retrieval_hit(data)
    except Exception as exc:  # noqa: BLE001 - 护栏失败回退 found 判断
        print(f"[warn] 检索护栏失败，回退 found 判断：{type(exc).__name__}: {exc}")
        retrieval = None if data.get("found") else {"rule": "legacy.no_evidence"}
    if retrieval:
        rails = [*(state.get("rails") or [])]
        rails.append(retrieval)
        return {
            "tool_results": tool_results,
            "reply": (
                f"我查了知识库，没有找到可靠的依据（{data.get('reason')}）。"
                "为避免给您错误信息，我帮您转接人工客服核实。"
            ),
            "handoff": True,
            "handoff_reason": f"知识库无据：{data.get('reason')}",
            "citations": [],
            "active_flow": "",
            "rails": rails,
        }

    top = data["hits"][0]
    # 技能注入点：默认 SKILL_ANSWER=off，走下面的规则模板；
    # 打开后由技能手册 + 知识库片段生成回复，引用仍然照常附上。
    composed = None
    if state.get("skill_prompt"):
        from agent.skills.compose import compose_answer

        composed = compose_answer(text, state["skill_prompt"], data.get("hits") or [])
    if composed:
        lines = [composed, "", "参考来源："]
    else:
        lines = [
            f"根据《{top['doc_title']}》（{top['section']}）：",
            str(top["text"]),
            "",
            "参考来源：",
        ]
    for index, citation in enumerate(data["citations"], start=1):
        lines.append(
            f"{index}. 《{citation['title']}》{citation['section']}｜{citation['source']}"
        )
    return {
        "tool_results": tool_results,
        "reply": "\n".join(lines),
        "citations": data["citations"],
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
    }


# 分支名 -> 节点函数。放在模块末尾注册，避免"函数还没定义就引用"的顺序问题。
MULTI_BRANCH_NODES.update(
    {
        "order": order_node,
        "claim": claim_node,
        "faq": faq_node,
        "handoff": handoff_node,
        "fallback": fallback_node,
    }
)
