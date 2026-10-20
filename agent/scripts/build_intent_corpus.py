"""合并多路语料、归一化、去重，产出待标注语料。

三路来源：
1. 模板/人工草稿：data/intent/seeds/seed_questions.jsonl
2. 检索评测集问句：eval/datasets/qa_golden.jsonl（当种子用）
3. 外部公开语料：data/intent/external/ 下的文件（如 JDDC），格式宽松解析

用法（在 agent/ 目录下执行）：
    python scripts/build_intent_corpus.py
    python scripts/build_intent_corpus.py --inspect-external     # 只看外部语料能解析出什么
    python scripts/build_intent_corpus.py --similarity 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402

SEEDS_PATH = DATA_DIR / "intent" / "seeds" / "seed_questions.jsonl"
EXTERNAL_DIR = DATA_DIR / "intent" / "external"
RAW_DIR = DATA_DIR / "intent" / "raw"
CORPUS_PATH = RAW_DIR / "corpus_v1.jsonl"
MULTITURN_PATH = RAW_DIR / "multiturn_v1.jsonl"
GOLD_PATH = AGENT_ROOT / "eval" / "datasets" / "qa_golden.jsonl"

USER_FIELD_CANDIDATES = ("text", "content", "utterance", "query", "question", "user_input")
USER_ROLE_VALUES = {"user", "customer", "buyer", "客户", "用户", "消费者"}

# JDDC chat.txt 的表头：session_id  user_id  waiter_send  is_transfer  is_repeat  content
JDDC_REQUIRED_COLUMNS = ("session_id", "content")
JDDC_SENDER_COLUMN = "waiter_send"

# 脱敏占位符（如 [地址x]、[数字x]）与纯符号碎片
PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]{1,12}\]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
MAX_PLACEHOLDER_RATIO = 0.4

# 对话碎片/礼貌用语：没有独立意图，短句里出现就直接丢
POLITE_FRAGMENTS = (
    "谢谢", "好的", "好吧", "好了", "没有了", "没了", "我懂了", "知道了", "明白",
    "嗯", "哦", "行吧", "可以了", "我试试", "不用了", "不用", "没事", "其他没有",
)
POLITE_FRAGMENT_MAX_LENGTH = 12


def is_noise(text: str, *, min_length: int = 3) -> bool:
    """剔除明显噪音：过短、没有中文、或占位符占比过高。"""
    cleaned = normalize(text)
    if len(cleaned) < min_length:
        return True
    if not CJK_PATTERN.search(cleaned):
        return True
    placeholder_chars = sum(len(match.group(0)) for match in PLACEHOLDER_PATTERN.finditer(cleaned))
    if placeholder_chars / len(cleaned) > MAX_PLACEHOLDER_RATIO:
        return True
    # 短句里含礼貌/收尾用语 → 属于对话过程语，没有独立意图
    if len(cleaned) <= POLITE_FRAGMENT_MAX_LENGTH and any(
        fragment in cleaned for fragment in POLITE_FRAGMENTS
    ):
        return True
    return False


def normalize(text: str) -> str:
    """统一空白、去掉首尾标点噪音，保留原始语义。"""
    cleaned = " ".join(str(text).split())
    return cleaned.strip("，,。.！!？?；; ")


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)} or {text}


def dedupe(rows: list[dict], similarity: float = 0.9) -> tuple[list[dict], int]:
    """精确去重 + 相似度去重（字符 bigram Jaccard），返回 (保留, 去掉条数)。

    去重按 (文本, 上下文) 判定：同一句话配不同上下文（如多轮样本）要各自保留。
    """
    kept: list[dict] = []
    seen_exact: set[str] = set()
    buckets: dict[tuple[str, str], list[tuple[str, set[str]]]] = {}
    removed = 0

    for row in rows:
        text = row["text"]
        if not text:
            removed += 1
            continue
        context_key = json.dumps(row.get("context"), ensure_ascii=False, sort_keys=True)
        exact_key = f"{text}|{context_key}"
        if exact_key in seen_exact:
            removed += 1
            continue

        grams = _bigrams(text)
        bucket = buckets.setdefault((text[0], context_key), [])
        duplicate = False
        for other_text, other_grams in bucket:
            if abs(len(other_text) - len(text)) > 4:
                continue
            union = grams | other_grams
            if union and len(grams & other_grams) / len(union) >= similarity:
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue

        seen_exact.add(exact_key)
        bucket.append((text, grams))
        kept.append(row)
    return kept, removed


def assign_ids(rows: list[dict]) -> list[dict]:
    """给每条语料分配稳定 id：同一句话（含上下文）重建后 id 不变，断点续跑才可靠。"""
    identified: list[dict] = []
    for row in rows:
        signature = row["text"] + "|" + json.dumps(row.get("context"), ensure_ascii=False, sort_keys=True)
        digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
        identified.append({**row, "id": f"c-{digest}"})
    return identified


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_seeds(path: Path = SEEDS_PATH) -> list[dict]:
    return [
        {
            "text": normalize(row["text"]),
            "context": row.get("context"),
            "hint_labels": row.get("hint_labels") or [],
            "source": row.get("source", "seed"),
        }
        for row in load_jsonl(path)
    ]


def load_gold_questions(path: Path = GOLD_PATH) -> list[dict]:
    rows = []
    for item in load_jsonl(path):
        rows.append(
            {
                "text": normalize(item["question"]),
                "context": item.get("context"),
                "hint_labels": [],
                "source": "retrieval_golden",
            }
        )
    return rows


def load_multiturn(path: Path = MULTITURN_PATH) -> list[dict]:
    """多轮样本（由 build_multiturn_samples.py 生成），带上文一起进语料。"""
    rows = []
    for item in load_jsonl(path):
        rows.append(
            {
                "text": normalize(item["text"]),
                "context": item.get("context"),
                "hint_labels": item.get("hint_labels") or [],
                "source": item.get("source", "jddc_multiturn"),
                "session_id": item.get("session_id"),
            }
        )
    return rows


def _iter_user_texts(obj, out: list[str]) -> None:
    """宽松地从各种公开语料结构里抽取"用户说过的话"。"""
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, list):
        for item in obj:
            _iter_user_texts(item, out)
        return
    if not isinstance(obj, dict):
        return

    role = str(
        obj.get("role") or obj.get("author") or obj.get("speaker") or obj.get("from") or ""
    ).lower()
    if role in USER_ROLE_VALUES:
        for field in USER_FIELD_CANDIDATES:
            value = obj.get(field)
            if isinstance(value, str) and value.strip():
                out.append(value)
                return

    for key in ("dialog", "dialogs", "session", "sessions", "conversation", "messages", "data"):
        if key in obj:
            _iter_user_texts(obj[key], out)
            return


def _split_row(line: str) -> list[str]:
    """同时兼容 tab 分隔与多空格分隔。"""
    if "\t" in line:
        return [part.strip() for part in line.split("\t")]
    return [part.strip() for part in re.split(r"\s{2,}", line.strip())]


def _parse_dialogue_tsv(text: str) -> list[dict]:
    """解析 JDDC 风格的会话 TSV：跳过表头，只保留用户侧发言，并带回 session_id。"""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header = _split_row(lines[0])
    if not all(column in header for column in JDDC_REQUIRED_COLUMNS):
        return []

    index = {column: header.index(column) for column in header}
    sender_index = index.get(JDDC_SENDER_COLUMN)
    content_index = index["content"]
    separator = "\t" if "\t" in lines[0] else " "
    rows: list[dict] = []
    for line in lines[1:]:
        parts = _split_row(line)
        if len(parts) <= content_index:
            continue
        # 真实文件里 content 之前可能夹着空列，且内容本身可能含分隔符：
        # 从表头里的 content 位置一直拼到行尾，再 strip，兼容两种写法。
        content = separator.join(parts[content_index:]).strip()
        if not content:
            continue
        # waiter_send = 1 表示客服/机器人发的，只要用户说的话
        if sender_index is not None and parts[sender_index] not in {"0", ""}:
            continue
        rows.append({"text": content, "session_id": parts[index["session_id"]]})
    return rows

    # 没有角色信息时，退一步取文本字段
    for field in USER_FIELD_CANDIDATES:
        value = obj.get(field)
        if isinstance(value, str) and value.strip():
            out.append(value)
            return


def load_external(
    directory: Path = EXTERNAL_DIR,
    *,
    inspect: bool = False,
    min_length: int = 3,
) -> list[dict]:
    """读取外部语料目录；支持会话 TSV（JDDC chat.txt）、.json / .jsonl / .txt。"""
    if not directory.exists():
        if inspect:
            print(f"外部语料目录不存在：{directory}")
        return []

    rows: list[dict] = []
    stats: list[tuple[str, int, int]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        parsed: list[dict] = []
        try:
            if suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        texts: list[str] = []
                        _iter_user_texts(json.loads(line), texts)
                        parsed.extend({"text": text, "session_id": None} for text in texts)
            elif suffix == ".json":
                texts = []
                _iter_user_texts(json.loads(path.read_text(encoding="utf-8")), texts)
                parsed.extend({"text": text, "session_id": None} for text in texts)
            elif suffix == ".txt":
                content = path.read_text(encoding="utf-8")
                tsv_rows = _parse_dialogue_tsv(content)
                if tsv_rows:
                    parsed.extend(tsv_rows)
                else:  # 纯文本：一行一句
                    parsed.extend(
                        {"text": line.strip(), "session_id": None}
                        for line in content.splitlines()
                        if line.strip()
                    )
            else:
                continue
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[warn] 跳过 {path.name}：{exc}")
            continue

        raw_count = len(parsed)
        kept = [item for item in parsed if not is_noise(item["text"], min_length=min_length)]
        stats.append((path.name, raw_count, len(kept)))

        if inspect:
            print(
                f"  {path.name}: 解析出 {raw_count} 条用户语句，"
                f"剔除短句/占位符噪音后 {len(kept)} 条"
            )
            for item in kept[:3]:
                sample = item["text"]
                print(f"      - {sample[:60]}")

        rows.extend(
            {
                "text": normalize(item["text"]),
                "context": None,
                "hint_labels": [],
                "source": f"external:{path.name}",
                "session_id": item.get("session_id"),
            }
            for item in kept
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="构建意图标注语料")
    parser.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    parser.add_argument("--similarity", type=float, default=0.9, help="相似度去重阈值")
    parser.add_argument(
        "--external-limit", type=int, default=3000, help="外部语料最多采样多少条（0 表示全部）"
    )
    parser.add_argument("--external-min-length", type=int, default=3, help="外部语料最短长度（字符）")
    parser.add_argument("--seed", type=int, default=42, help="外部语料抽样随机种子")
    parser.add_argument("--inspect-external", action="store_true", help="只检查外部语料解析结果")
    args = parser.parse_args()

    if args.inspect_external:
        print(f"检查外部语料：{args.external_dir}")
        load_external(args.external_dir, inspect=True, min_length=args.external_min_length)
        return

    seeds = load_seeds(args.seeds)
    gold = load_gold_questions()
    multiturn = load_multiturn()
    external_all = load_external(args.external_dir, min_length=args.external_min_length)
    external = external_all
    if args.external_limit and len(external_all) > args.external_limit:
        random.seed(args.seed)
        external = random.sample(external_all, args.external_limit)

    merged = seeds + gold + multiturn + external
    kept, removed = dedupe(merged, similarity=args.similarity)
    kept = assign_ids(kept)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n",
        encoding="utf-8",
    )

    print(
        f"语料来源：草稿 {len(seeds)} 条，检索评测集 {len(gold)} 条，"
        f"多轮样本 {len(multiturn)} 条，外部语料 {len(external_all)} 条（按上限抽样 {len(external)} 条）"
    )
    print(f"合并 {len(merged)} 条，去重后 {len(kept)} 条（去掉 {removed} 条）")
    source_counts = Counter(row["source"].split(":")[0] for row in kept)
    print("来源分布：" + "，".join(f"{key} {value}" for key, value in source_counts.most_common()))
    with_hint = sum(1 for row in kept if row["hint_labels"])
    with_context = sum(1 for row in kept if row.get("context"))
    with_session = sum(1 for row in kept if row.get("session_id"))
    print(f"带提示标签 {with_hint} 条，带上下文 {with_context} 条，带 session_id {with_session} 条")
    print(f"已写入：{args.output}")


if __name__ == "__main__":
    main()
