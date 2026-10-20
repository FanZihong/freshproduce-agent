"""知识库反哺：把"知识库答不出来"的真实问题补成条目，重建索引，再跑检索回归。

三步，人工在中间：
1. `--seed`：从 trace / 候选池里挑出"无据转人工"的轮次，生成待填的条目骨架
   （`eval/datasets/kb_candidates.jsonl`，只填对了答案的行才会被写入知识库）；
2. 人工打开骨架，把 `answer` 写实（顺便核对 `doc` / `section` 选得对不对）；
3. `--apply`：把填好的条目插进 `data/knowledge/*.md` 对应小节的末尾，
   备份原文件 → 重建索引 → 打印回归命令。

为什么不做全自动：知识库条目的正确性只能由人保证。脚本能做的只是"找到该补什么"、
"插到正确的位置"、"别把文件写坏"（改前自动备份、重复条目跳过、幂等）。

**默认追加到文件末尾（`--append-mode end`），不插进中间小节**——这是被门禁打回来过一次的教训：
规则分块的 chunk id 是"文件内全局序号"（`售后政策总则#rule#005`），
往中间插一条，后面所有条目的编号就整体后移，检索黄金集里按 id 写的期望值全部错位。
实测过一次：往"受理范围"插了 1 条，`退款与退货` 的无理由退货条目从 #005 变 #006，
q017/q018 的期望 id 直接指错，q019/q020 各丢半条 → recall@5 从 0.8333 掉到 0.8000，
看起来像检索退化，其实内容一条没变。追加到末尾则只新增编号，已有条目 id 稳定。
确实需要按主题插进中间时，用 `--append-mode section`，但要记得黄金集的期望 id 得跟着重标。

用法（在 agent/ 目录下执行）：
    python scripts/update_knowledge.py --seed                 # 生成/更新骨架
    python scripts/update_knowledge.py --dry-run              # 看看会插到哪（默认就是 dry-run）
    python scripts/update_knowledge.py --apply                # 真写 + 重建索引
    python scripts/update_knowledge.py --probe "生鲜能开发票吗"  # 先问一句，看知识库答不答得上
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.trace import DEFAULT_TRACE_DIR, load_traces  # noqa: E402

DEFAULT_KNOWLEDGE_DIR = DATA_DIR / "knowledge"
DEFAULT_POOL = AGENT_ROOT / "eval" / "datasets" / "trace_badcase_pool.jsonl"
DEFAULT_CANDIDATES = AGENT_ROOT / "eval" / "datasets" / "kb_candidates.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "knowledge_update.md"

# 追加模式下新条目的落点：文件末尾的"补充问答"小节
EXTRAS_SECTION = "补充问答"

# "知识库无据"的典型话术：命中这些说明系统当场承认答不上来
NO_EVIDENCE_MARKS: tuple[str, ...] = (
    "没有找到可靠的依据",
    "知识库无据",
    "没有找到可靠的",
)

# 新条目该进哪个文件、哪个小节：按关键词给建议，人工可以改
SECTION_HINTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("退款", "退货", "到账", "退钱"), "常见问题FAQ.md", "3. 退款与发票"),
    (("发票", "开票", "抬头", "税号"), "常见问题FAQ.md", "3. 退款与发票"),
    (("赔付", "赔", "坏", "烂", "理赔"), "常见问题FAQ.md", "2. 品质与赔付"),
    (("发货", "物流", "签收", "配送", "快递", "到哪"), "常见问题FAQ.md", "1. 订单与物流"),
    (("保存", "储存", "冷藏", "冷冻", "保鲜", "放多久"), "常见问题FAQ.md", "4. 商品与储存"),
    (("催单", "补偿", "超时"), "物流时效说明.md", "4. 催单与补偿"),
    (("缺斤少两", "重量", "称重", "分量"), "赔付标准表.md", "3. 缺斤少两赔付"),
    (("转人工", "人工客服", "投诉"), "售后政策总则.md", "4. 转人工规则"),
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


# ---------------- 纯函数：判定、建议、插入 ----------------


def needs_knowledge(*texts: str | None) -> bool:
    """这些文本里有没有"我答不上来"的痕迹。"""
    return any(mark in (text or "") for text in texts for mark in NO_EVIDENCE_MARKS)


def suggest_target(question: str, default_doc: str = "常见问题FAQ.md") -> tuple[str, str]:
    """按关键词猜该补进哪个文件、哪个小节；猜不出来就给默认文件、空小节。"""
    text = str(question or "")
    for keywords, doc, section in SECTION_HINTS:
        if any(word in text for word in keywords):
            return doc, section
    return default_doc, ""


def normalize_question(question: str) -> str:
    """统一成"以问号结尾"的形式，避免渲染出"问：xxx答：yyy"。"""
    text = str(question or "").strip()
    if not text:
        return ""
    return text if text[-1] in "？?" else text + "？"


def render_entry(index: int, question: str, answer: str) -> str:
    return f"{index}. 问：{normalize_question(question)}答：{str(answer).strip()}"


def section_bounds(markdown: str, section: str) -> tuple[int, int] | None:
    """找小节的范围：返回 (标题行起始下标, 该小节内容结束下标)；找不到返回 None。"""
    lines = markdown.splitlines(keepends=True)
    title = f"## {section}".strip()
    start = None
    offset = 0
    start_offset = 0
    for line in lines:
        stripped = line.strip()
        if start is None and stripped == title:
            start = True
            start_offset = offset
        elif start is not None and stripped.startswith("## "):
            return (start_offset, offset)
        offset += len(line)
    if start is not None:
        return (start_offset, len(markdown))
    return None


def next_index_in_section(markdown: str, section: str) -> int:
    """小节里已有的编号最大值 + 1（找不到小节就从 1 开始）。"""
    bounds = section_bounds(markdown, section)
    if bounds is None:
        return 1
    body = markdown[bounds[0] : bounds[1]]
    highest = 0
    for line in body.splitlines():
        head = line.strip().split(".", 1)[0]
        if head.isdigit():
            highest = max(highest, int(head))
    return highest + 1


def insert_entry(markdown: str, section: str, entry: str) -> tuple[str, bool]:
    """把条目插到小节末尾；小节不存在时返回原样 + False（由调用方决定怎么办）。"""
    bounds = section_bounds(markdown, section)
    if bounds is None:
        return markdown, False
    start, end = bounds
    body = markdown[start:end]
    trailing = "\n" if body.endswith("\n") else "\n\n"
    if not body.endswith("\n"):
        trailing = "\n"
    block = body.rstrip("\n") + "\n" + entry + "\n"
    if end < len(markdown) and not markdown[end:].startswith("\n"):
        # 后面紧跟下一个 "## " 标题时补一个空行，保持 markdown 结构
        block += "\n"
    return markdown[:start] + block + markdown[end:], True


def append_new_section(markdown: str, title: str, entry: str) -> str:
    """小节不存在时，在文件末尾开一个补充小节。"""
    body = markdown.rstrip("\n")
    return f"{body}\n\n## {title}\n\n{entry}\n"


def entry_exists(markdown: str, question: str) -> bool:
    """幂等检查：这条问题是不是已经在文件里了（按问句主体比对，忽略标点差异）。"""
    core = normalize_question(question).rstrip("？?").strip()
    if not core:
        return False
    return core in markdown


def merge_candidates(existing: list[dict], new_rows: list[dict]) -> tuple[list[dict], int]:
    """按问题合并候选条目：已经填过答案的行不会被空骨架覆盖。"""
    by_question = {normalize_question(row.get("question")): row for row in existing}
    added = 0
    for row in new_rows:
        key = normalize_question(row.get("question"))
        if not key:
            continue
        if key in by_question:
            old = by_question[key]
            if not old.get("answer") and row.get("answer"):
                old["answer"] = row["answer"]
            continue
        by_question[key] = row
        added += 1
    return list(by_question.values()), added


# ---------------- 打包：从 trace / 池子挑"该补什么" ----------------


def candidates_from_pool(pool_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in pool_rows:
        observed = row.get("observed") or {}
        if not needs_knowledge(observed.get("reply"), observed.get("handoff_reason")):
            continue
        doc, section = suggest_target(row.get("text") or "")
        rows.append(
            {
                "id": f"kb-{row.get('id')}",
                "question": row.get("text") or "",
                "answer": "",
                "doc": doc,
                "section": section,
                "source": f"pool:{row.get('id')}",
                "observed_reply": (observed.get("reply") or "")[:120],
                "note": "",
                "status": "pending",
            }
        )
    return rows


def candidates_from_traces(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if not needs_knowledge(record.get("reply"), record.get("handoff_reason")):
            continue
        doc, section = suggest_target(record.get("user_input") or "")
        rows.append(
            {
                "id": f"kb-{record.get('id') or record.get('session_id')}-t{record.get('turn')}",
                "question": record.get("user_input") or "",
                "answer": "",
                "doc": doc,
                "section": section,
                "source": f"trace:{record.get('session_id')}#{record.get('turn')}",
                "observed_reply": (record.get("reply") or "")[:120],
                "note": "",
                "status": "pending",
            }
        )
    return rows


# ---------------- 执行 ----------------


def run_seed(args: argparse.Namespace) -> int:
    pool_rows = load_jsonl(args.pool) if args.pool.exists() else []
    trace_records = load_traces(args.trace) if args.trace else []
    fresh = candidates_from_pool(pool_rows) + candidates_from_traces(trace_records)
    existing = load_jsonl(args.candidates)
    merged, added = merge_candidates(existing, fresh)

    print(f"候选池：{args.pool}（{len(pool_rows)} 条）")
    print(f"trace：{args.trace or '（未指定）'}（{len(trace_records)} 轮）")
    print(f"发现「知识库无据」的样本：{len(fresh)} 条；合并后 {len(merged)} 条（新增 {added} 条）")
    if merged:
        write_jsonl(args.candidates, merged)
        print(f"待填骨架已写入：{args.candidates}")
        print("下一步：打开它把 answer 写实（doc/section 也可改），然后跑 --apply")
    else:
        print(
            "本次没有发现'知识库无据'的样本——说明现有对话里没有问到知识库覆盖不到的坑。\n"
            "  想主动补条目的话：把你关心的问题直接写进候选文件，或用 --probe 先问问看"
        )
    return 0


def run_probe(args: argparse.Namespace) -> int:
    """先用本地检索问一句：知识库答不答得上（不调大模型，纯检索）。"""
    from agent.config import get_env
    from agent.tools.faq import search_faq, get_retriever

    retriever = get_retriever(reranker_kind=get_env("FAQ_RERANK", "score") or "score")
    print(f"知识库：{len(retriever.chunks)} 个 chunk")
    for question in args.probe:
        result = search_faq(question)
        verdict = "有据" if result["found"] else "无据（该补条目）"
        print(f"\n问：{question}\n  判定：{verdict} | {result['reason']}")
        if result["found"]:
            top = (result.get("hits") or [{}])[0]
            print(f"  命中：{top.get('doc_title')} / {top.get('section')}")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.candidates)
    if not rows:
        raise SystemExit(f"候选文件是空的：{args.candidates}（先跑 --seed 或手写几条）")
    ready = [
        row
        for row in rows
        if str(row.get("answer") or "").strip() and str(row.get("question") or "").strip()
    ]
    pending = len(rows) - len(ready)
    print(f"候选 {len(rows)} 条：填好答案 {len(ready)} 条、待填 {pending} 条")
    if not ready:
        raise SystemExit("没有填好 answer 的条目，先补答案再 --apply")

    changes: dict[Path, list[dict]] = {}
    for row in ready:
        doc = str(row.get("doc") or "常见问题FAQ.md")
        path = Path(args.knowledge_dir) / doc
        if not path.exists():
            print(f"  [!] 跳过 {row['id']}：找不到 {path}")
            continue
        changes.setdefault(path, []).append(row)

    summary: list[str] = []
    for path, items in changes.items():
        markdown = path.read_text(encoding="utf-8")
        inserted: list[str] = []
        for row in items:
            section = str(row.get("section") or "")
            if entry_exists(markdown, row["question"]):
                summary.append(f"- {row['id']}：问题已存在，跳过")
                continue
            if args.append_mode == "end":
                # 追加到文件末尾：只新增编号，已有条目的 chunk id 不动（黄金集不会错位）
                target_section = EXTRAS_SECTION
                index = next_index_in_section(markdown, target_section)
                entry = render_entry(index, row["question"], row["answer"])
                if section_bounds(markdown, target_section) is None:
                    markdown = append_new_section(markdown, target_section, entry)
                    summary.append(
                        f"- {row['id']}：追加到 {path.name} 末尾（新建小节「{target_section}」，第 {index} 条）"
                    )
                else:
                    markdown, _ = insert_entry(markdown, target_section, entry)
                    summary.append(
                        f"- {row['id']}：追加到 {path.name} / {target_section}（第 {index} 条）"
                    )
            else:
                # 按主题插进中间小节：可读性更好，但会移动后面所有条目的编号
                target_section = section or suggest_target(row["question"])[1]
                index = next_index_in_section(markdown, target_section) if target_section else 1
                entry = render_entry(index, row["question"], row["answer"])
                if target_section and section_bounds(markdown, target_section) is not None:
                    markdown, _ = insert_entry(markdown, target_section, entry)
                    summary.append(
                        f"- {row['id']}：插入 {path.name} / {target_section}（第 {index} 条，"
                        "注意：后面的 chunk id 会整体后移，黄金集期望 id 需要重标）"
                    )
                else:
                    index = next_index_in_section(markdown, EXTRAS_SECTION)
                    entry = render_entry(index, row["question"], row["answer"])
                    markdown = append_new_section(markdown, EXTRAS_SECTION, entry)
                    summary.append(
                        f"- {row['id']}：小节「{target_section or '（未指定）'}」不存在，"
                        f"已追加到 {path.name} / 「{EXTRAS_SECTION}」（第 {index} 条）"
                    )
            inserted.append(entry)
        if inserted and not args.dry_run:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text(markdown, encoding="utf-8")
            print(f"已写入：{path}（备份：{backup.name}）")

    print("\n变更明细：")
    print("\n".join(summary) if summary else "（无）")

    if args.dry_run:
        print("\n这是 dry-run：文件没动。确认无误后加 --apply 真写。")
        return 0

    if not args.no_rebuild:
        print("\n重建索引…")
        completed = subprocess.run(
            [sys.executable, str(AGENT_ROOT / "scripts" / "build_knowledge.py")],
            cwd=str(AGENT_ROOT),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        tail = (completed.stdout or "").strip().splitlines()[-3:]
        print("\n".join("  " + line for line in tail))
        if completed.returncode != 0:
            print("  [!] 索引重建失败：")
            print("\n".join("  " + line for line in (completed.stderr or "").splitlines()[-5:]))
            return 1

    report = [
        "# 知识库反哺记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 候选文件：{args.candidates}",
        f"- 本次写入 {len(ready)} 条（待填 {pending} 条）",
        "",
        *summary,
        "",
        "## 下一步",
        "",
        "```bash",
        "python scripts/regression.py --only retrieval",
        "```",
        "",
        "验收：检索指标不劣化；补了条目的那几个问题，`search_faq` 应该从「无据」变成「有据」。",
        "",
    ]
    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"\n记录已写入：{args.report}")
    print("\n下一步：python scripts/regression.py --only retrieval")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库反哺：补条目 → 重建索引 → 检索回归")
    parser.add_argument("--seed", action="store_true", help="从 trace/池子挑出'答不上来'的问题生成骨架")
    parser.add_argument("--apply", action="store_true", help="把填好的条目写进知识库并重建索引")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入的内容（默认行为）")
    parser.add_argument("--probe", nargs="*", default=None, help="先问几句，看知识库答不答得上")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--append-mode",
        choices=["end", "section"],
        default="end",
        help="end=追加到文件末尾的「补充问答」（默认，不动已有条目的编号）；"
        "section=按主题插进中间小节（可读性好，但会移动后面条目的 chunk id）",
    )
    parser.add_argument("--no-rebuild", action="store_true", help="不重建索引（测试用）")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.probe is not None:
        return run_probe(args)
    if args.apply or args.dry_run:
        return run_apply(args)
    return run_seed(args)


if __name__ == "__main__":
    raise SystemExit(main())
