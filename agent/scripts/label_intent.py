"""大模型批量预标注意图标签。

特性：并发调用、指数退避、断点续跑（已标注的跳过）、每 N 条落盘、离线 dry-run。

用法（在 agent/ 目录下执行）：
    python scripts/label_intent.py --dry-run                 # 离线验证流水线，不消耗 API
    python scripts/label_intent.py --concurrency 8            # 真实调用（读取 .env 的 LLM_API_KEY）
    python scripts/label_intent.py --limit 100 --dry-run      # 小批量试跑
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.config import get_env  # noqa: E402
from agent.db import DATA_DIR  # noqa: E402
from agent.intent.classifier import classify  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
CORPUS_PATH = RAW_DIR / "corpus_v1.jsonl"
OUTPUT_PATH = RAW_DIR / "prelabel_v1.jsonl"


def build_system_prompt() -> str:
    """把 taxonomy 渲染成紧凑的标注指令。"""
    taxonomy = load_taxonomy()
    lines = [
        "你是生鲜电商售后客服对话的意图标注员。",
        f"只能从下面 {len(taxonomy.labels)} 个标签中选择，最多 2 个；多意图时按重要性排序（动作型优先）。",
        "",
    ]
    for spec in taxonomy.labels:
        lines.append(f"- {spec.key}（{spec.name}）：{spec.definition}")
        if spec.confusions:
            tip = spec.confusions[0].tip
            lines.append(f"    边界提示：{tip}")

    guidelines = taxonomy.labeling_guidelines[:4]
    if guidelines:
        lines.append("")
        lines.append("标注规则：")
        lines.extend(f"- {rule}" for rule in guidelines)

    lines += [
        "",
        "输出要求：只输出一个 JSON，不要解释、不要代码块，格式为",
        '{"labels": ["标签1"], "confidence": 0.0, "reason": "一句话理由"}',
        "confidence 是你对这次标注的把握（0-1）；完全超出业务范围时用 chitchat。",
    ]
    return "\n".join(lines)


def parse_label_response(raw: str, valid_labels: set[str]) -> tuple[list[str], float, str]:
    """解析模型输出；返回 (labels, confidence, reason)，解析失败抛 ValueError。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON：{raw[:80]}")
    payload = json.loads(text[start : end + 1])

    labels = [str(item) for item in (payload.get("labels") or []) if str(item) in valid_labels]
    if not labels:
        raise ValueError(f"没有合法标签：{payload.get('labels')}")
    confidence = float(payload.get("confidence") or 0.0)
    reason = str(payload.get("reason") or "")[:120]
    return labels[:2], max(0.0, min(1.0, confidence)), reason


def dry_run_label(row: dict, valid_labels: set[str]) -> tuple[list[str], float, str]:
    """离线标注：优先用草稿里的提示标签，否则退回规则版（仅用于验证流水线）。"""
    hints = [label for label in row.get("hint_labels") or [] if label in valid_labels]
    if hints:
        return hints, 0.99, "dry-run：使用草稿提示标签"
    mapped = classify(row["text"])
    taxonomy = load_taxonomy()
    label = taxonomy.spec(mapped.label).key if taxonomy.spec(mapped.label) else "chitchat"
    return [label], 0.5, f"dry-run：规则版判为 {mapped.label}"


def _call_llm(system_prompt: str, row: dict, model: str, timeout: float) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=timeout,
    )
    user_parts = [f"用户当前问句：{row['text']}"]
    context = row.get("context") or {}
    if context.get("last_question"):
        user_parts.append(f"上一轮用户问题：{context['last_question']}")
    if context.get("product"):
        user_parts.append(f"上文商品：{context['product']}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="大模型批量预标注意图")
    parser.add_argument("--input", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--concurrency", type=int, default=int(get_env("LABEL_CONCURRENCY", "6")))
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条（0 表示全部）")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--model", default=get_env("LABEL_MODEL", "deepseek-chat"))
    parser.add_argument("--dry-run", action="store_true", help="离线跑通流水线，不调用 API")
    parser.add_argument("--flush-every", type=int, default=20)
    args = parser.parse_args()

    corpus = load_jsonl(args.input)
    if not corpus:
        print(f"没有找到待标注语料：{args.input}（先跑 scripts/build_intent_corpus.py）")
        return

    existing = load_jsonl(args.output)
    done_ids = {row["id"] for row in existing}
    pending = [row for row in corpus if row["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    taxonomy = load_taxonomy()
    valid_labels = set(taxonomy.keys)
    system_prompt = build_system_prompt()

    print(f"待标注 {len(pending)} 条（已完成 {len(done_ids)} 条，语料共 {len(corpus)} 条）")
    if not pending:
        print("没有需要处理的新样本。")
        return
    print(f"模式：{'dry-run（离线）' if args.dry_run else f'LLM {args.model}'}，并发 {args.concurrency}")

    if not args.dry_run:
        api_key = get_env("LLM_API_KEY", "")
        if not api_key or "替换" in api_key:
            print("[错误] agent/.env 里的 LLM_API_KEY 还没配置（当前是占位符）。")
            print("       填好真实 key 后重跑；想先验证流程可以加 --dry-run。")
            return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    counter = 0

    def handle(row: dict) -> dict:
        if args.dry_run:
            labels, confidence, reason = dry_run_label(row, valid_labels)
            return {"labels": labels, "confidence": confidence, "reason": reason}
        raw = _call_llm(system_prompt, row, args.model, args.timeout)
        labels, confidence, reason = parse_label_response(raw, valid_labels)
        return {"labels": labels, "confidence": confidence, "reason": reason}

    with args.output.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(handle, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001 - 失败样本留待下次重跑
                    failures.append((row["id"], f"{type(exc).__name__}: {exc}"[:160]))
                    continue
                record = {
                    "id": row["id"],
                    "text": row["text"],
                    "context": row.get("context"),
                    "labels": payload["labels"],
                    "confidence": payload["confidence"],
                    "reason": payload["reason"],
                    "source": row.get("source"),
                    "labeler": "dry-run" if args.dry_run else f"llm:{args.model}",
                }
                with write_lock:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counter += 1
                    if counter % max(1, args.flush_every) == 0:
                        sink.flush()
                        print(f"  已写入 {counter}/{len(pending)} 条")
                    if counter % 200 == 0:
                        time.sleep(0.05)  # 轻微限速，避免触发上游限流
        sink.flush()

    labeled = load_jsonl(args.output)
    counts = Counter(label for row in labeled for label in row["labels"])
    low_conf = sum(1 for row in labeled if row.get("confidence", 1) < 0.6)
    print(f"\n预标注完成：累计 {len(labeled)} 条（本次新增 {counter} 条）")
    print(f"本轮失败 {len(failures)} 条（下次运行会自动重试）")
    print(f"低置信度（<0.6）{low_conf} 条，需要重点人工校准")
    print("标签分布：" + "，".join(f"{key} {value}" for key, value in counts.most_common()))
    print(f"输出文件：{args.output}")
    for sample_id, error in failures[:5]:
        print(f"  失败样例 {sample_id}: {error}")


if __name__ == "__main__":
    main()
