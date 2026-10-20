"""置信度校准与拒答决策：把"模型说 0.8"变成"真的 80% 对"，再据此决定放行/澄清/转人工。

为什么不是标准 temperature scaling：
分类头的 temperature scaling 作用在 logits 上（softmax(logits/T)）。而我们的置信度来自
**生成序列的几何平均 token 概率**，没有 10 类 logits。所以这里把温度作用在 **log-odds** 上：

    p_cal = sigmoid(logit(p) / T)

T > 1 会把过高的置信度压下来（微调模型几乎总是过度自信），T < 1 则相反。
等 Day 3 用上 `score_labels()` 的候选分布时，可以直接换成经典 softmax(logits/T)，
两条路的校准目标函数（NLL）是一样的。

阈值不是拍脑袋定的：`choose_thresholds` 在"拒答率 ≤ 上限"的约束下，最大化
**被拒样本的错误率 / 被接受样本的错误率**——拒答必须真的把错的那批挑出来，否则阈值没意义。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "eval" / "reports" / "intent_calibration.json"
EPS = 1e-6


def logit(probability: float) -> float:
    clipped = min(1 - EPS, max(EPS, float(probability)))
    return math.log(clipped / (1 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def apply_temperature(confidence: float, temperature: float) -> float:
    """把原始置信度按温度重新标定。"""
    if temperature <= 0:
        return float(confidence)
    return round(sigmoid(logit(confidence) / temperature), 4)


def _nll(confidences: list[float], corrects: list[int], temperature: float) -> float:
    total = 0.0
    for confidence, correct in zip(confidences, corrects):
        probability = sigmoid(logit(confidence) / temperature)
        probability = min(1 - EPS, max(EPS, probability))
        total -= math.log(probability if correct else 1 - probability)
    return total / max(1, len(confidences))


def fit_temperature(
    confidences: list[float], corrects: list[int], low: float = 0.05, high: float = 20.0, steps: int = 400
) -> float:
    """在验证集上网格搜索温度 T，使 NLL 最小（微调模型的 T 通常 > 1）。"""
    if not confidences:
        return 1.0
    best_temperature, best_nll = 1.0, float("inf")
    # 先粗搜（对数均匀），再在最优点附近细搜一轮
    for index in range(steps):
        ratio = index / (steps - 1)
        temperature = math.exp(math.log(low) + ratio * (math.log(high) - math.log(low)))
        score = _nll(confidences, corrects, temperature)
        if score < best_nll:
            best_temperature, best_nll = temperature, score
    span = best_temperature * 0.15
    for index in range(101):
        temperature = max(low, best_temperature - span + index * (2 * span / 100))
        score = _nll(confidences, corrects, temperature)
        if score < best_nll:
            best_temperature, best_nll = temperature, score
    return round(best_temperature, 4)


def reliability_table(
    confidences: list[float], corrects: list[int], bins: int = 10
) -> list[dict]:
    """可靠性曲线：每个置信度区间里"平均置信度 vs 真实准确率"。

    校准良好的模型，这两列应当接近（gap 接近 0）。
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for confidence, correct in zip(confidences, corrects):
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index].append((confidence, correct))

    table: list[dict] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        average = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        table.append(
            {
                "bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                "count": len(bucket),
                "avg_confidence": round(average, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(average - accuracy, 4),
            }
        )
    return table


def expected_calibration_error(table: list[dict], total: int | None = None) -> float:
    """ECE：按样本数加权的 |置信度 − 准确率| 平均值。"""
    if not table:
        return 0.0
    denominator = total or sum(item["count"] for item in table)
    if not denominator:
        return 0.0
    return round(
        sum(item["count"] * abs(item["gap"]) for item in table) / denominator, 4
    )


def choose_thresholds(
    confidences: list[float],
    corrects: list[int],
    max_refusal_rate: float = 0.10,
    min_refused_error_rate: float = 0.30,
) -> dict:
    """在拒答率上限内，挑出"最能挑出错误样本"的阈值。

    返回值里带上 ratio（被拒样本错误率 / 被接受样本错误率），它是阈值是否有效的判据：
    ratio ≈ 1 说明拒答完全是随机的，这种阈值不该用。
    """
    if not confidences:
        return {
            "accept_threshold": 0.0,
            "refusal_rate": 0.0,
            "refused_error_rate": 0.0,
            "accepted_error_rate": 0.0,
            "ratio": 0.0,
            "max_refusal_rate": max_refusal_rate,
        }

    candidates = sorted(set(confidences))
    best: dict | None = None
    total = len(confidences)
    for threshold in candidates:
        refused = [index for index, value in enumerate(confidences) if value < threshold]
        refusal_rate = len(refused) / total
        if refusal_rate > max_refusal_rate or not refused:
            continue
        refused_error = sum(1 for index in refused if not corrects[index]) / len(refused)
        accepted = [index for index in range(total) if index not in set(refused)]
        if not accepted:
            continue
        accepted_error = sum(1 for index in accepted if not corrects[index]) / len(accepted)
        if refused_error < min_refused_error_rate and refused_error < 2 * accepted_error:
            continue
        ratio = refused_error / accepted_error if accepted_error > 0 else float("inf")
        candidate = {
            "accept_threshold": round(threshold, 4),
            "refusal_rate": round(refusal_rate, 4),
            "refused_error_rate": round(refused_error, 4),
            "accepted_error_rate": round(accepted_error, 4),
            "ratio": round(ratio, 3) if ratio != float("inf") else None,
            "max_refusal_rate": max_refusal_rate,
        }
        if best is None or (candidate["ratio"] or 0) > (best["ratio"] or 0):
            best = candidate
    return best or {
        "accept_threshold": 0.0,
        "refusal_rate": 0.0,
        "refused_error_rate": 0.0,
        "accepted_error_rate": 0.0,
        "ratio": 0.0,
        "max_refusal_rate": max_refusal_rate,
        "note": "在拒答率上限内没找到有效阈值，说明当前置信度区分度不足",
    }


@dataclass
class DecisionConfig:
    """拒答策略配置（默认值是先验，正式值应由验证集拟合得到）。"""

    temperature: float = 1.0
    accept_threshold: float = 0.6
    handoff_threshold: float = 0.35
    max_clarify_rounds: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DecisionConfig":
        known = {key: value for key, value in (payload or {}).items() if key in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Decision:
    action: str          # accept / clarify / handoff
    probability: float   # 校准后的置信度
    reason: str
    reply: str = ""
    metadata: dict = field(default_factory=dict)


CLARIFY_MENU = (
    "没完全明白您的意思，帮我确认一下您想处理哪件事：\n"
    "1) 查询订单或物流进度\n"
    "2) 咨询退款、发票等规则\n"
    "3) 反馈商品质量问题（坏果、缺斤少两）\n"
    "直接回复序号或说一句话都行。"
)


def clarify_question(top_label: str | None = None, label_name: str = "") -> str:
    """低置信度时的澄清话术；有 top1 猜测时先给一个"是不是这个"的选项。"""
    if top_label and label_name:
        return (
            f"我猜您想咨询的是「{label_name}」相关的问题，如果是，请补充一下细节；\n"
            "如果不是，也可以直接说您要办的事。\n\n" + CLARIFY_MENU
        )
    return CLARIFY_MENU


def decide(
    confidence: float,
    clarify_rounds: int = 0,
    config: DecisionConfig | None = None,
    top_label: str | None = None,
    label_name: str = "",
) -> Decision:
    """核心决策：放行 / 澄清 / 转人工。

    clarify_rounds 复用图状态里的 loop_count 护栏：连续澄清到上限还没问明白就转人工，
    避免"机器人反复反问"的死循环。
    """
    config = config or DecisionConfig()
    probability = apply_temperature(confidence, config.temperature)
    # 澄清上限是对话护栏：阈值在 data/rails/dialog.yaml，行为与旧的 >= max 完全一致
    try:
        from agent.rails.policy import clarify_hit

        cap = clarify_hit(clarify_rounds, config.max_clarify_rounds)
    except Exception as exc:  # noqa: BLE001 - 护栏失败回退旧阈值
        print(f"[warn] 澄清护栏失败，回退旧阈值：{type(exc).__name__}: {exc}")
        cap = (
            {"rule": "legacy.clarify_cap"}
            if clarify_rounds >= config.max_clarify_rounds
            else None
        )

    if probability >= config.accept_threshold:
        return Decision("accept", probability, "置信度高于放行阈值")
    if probability < config.handoff_threshold and cap:
        return Decision("handoff", probability, "置信度过低且澄清已达上限", metadata={"rail": cap})
    if cap:
        return Decision(
            "handoff",
            probability,
            f"连续澄清 {clarify_rounds} 次仍未明确，转人工",
            metadata={"rail": cap},
        )
    return Decision(
        "clarify",
        probability,
        f"置信度低于放行阈值（{probability:.2f} < {config.accept_threshold}）",
        reply=clarify_question(top_label, label_name),
        metadata={"clarify_rounds": clarify_rounds + 1},
    )


def save_calibration(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration(path: Path | str | None = None) -> dict:
    """读取校准结果；不存在时返回默认配置（不会因为文件缺失而崩）。"""
    path = Path(path) if path else DEFAULT_CALIBRATION_PATH
    if not path.exists():
        return {"config": DecisionConfig().to_dict(), "source": "default"}
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_calibration(path: Path | str | None = None) -> DecisionConfig:
    payload = load_calibration(path)
    return DecisionConfig.from_dict(payload.get("config") or {})
