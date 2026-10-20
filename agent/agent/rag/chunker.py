"""知识库分块。

两套策略：
- rule：按"规则条"切分，一条规则一个 chunk，适合赔付标准与 FAQ；
- paragraph：按"小节 + 段落"切分，超长时按句子窗口切分并保留重叠，适合政策正文。

每个 chunk 都有稳定的 chunk_id（doc_id#策略#序号），评测集与消融实验靠它对齐。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DOC_TITLE_PREFIX = "# "
SECTION_PREFIX = "## "
SOURCE_PREFIX = "> 来源："

ITEM_PATTERN = re.compile(r"^(?:\d+[.、)]|[-*])\s+(.*)$")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")

DEFAULT_MAX_CHARS = 280
DEFAULT_OVERLAP = 40


@dataclass(frozen=True)
class Chunk:
    """一个可检索的最小单元。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    strategy: str
    text: str
    source: str
    char_count: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    """解析后的知识库文档。"""

    doc_id: str
    title: str
    source: str
    path: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)


def parse_document(path: Path) -> Document:
    """解析 Markdown：一级标题当文档名，「> 来源：」当来源，二级标题当小节。"""
    title = path.stem
    source = ""
    sections: list[tuple[str, list[str]]] = []
    current_title = "前言"
    current_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(SECTION_PREFIX):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line[len(SECTION_PREFIX) :].strip()
            current_lines = []
            continue
        if line.startswith(DOC_TITLE_PREFIX):
            title = line[len(DOC_TITLE_PREFIX) :].strip()
            continue
        if line.startswith(SOURCE_PREFIX):
            source = line[len(SOURCE_PREFIX) :].strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return Document(
        doc_id=path.stem,
        title=title,
        source=source,
        path=str(path),
        sections=sections,
    )


def _strip_item_prefix(line: str) -> str:
    match = ITEM_PATTERN.match(line)
    return match.group(1).strip() if match else line


def _make_chunk(doc: Document, section: str, strategy: str, index: int, text: str) -> Chunk:
    cleaned = text.strip()
    return Chunk(
        chunk_id=f"{doc.doc_id}#{strategy}#{index:03d}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        section=section,
        strategy=strategy,
        text=cleaned,
        source=doc.source,
        char_count=len(cleaned),
    )


def chunk_by_rule(doc: Document) -> list[Chunk]:
    """策略一：一条规则一个 chunk（去掉序号）。"""
    chunks: list[Chunk] = []
    index = 0
    for section_title, lines in doc.sections:
        for line in lines:
            text = _strip_item_prefix(line)
            if len(text) < 4:
                continue
            index += 1
            chunks.append(_make_chunk(doc, section_title, "rule", index, text))
    return chunks


def _split_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    """按句子切窗口，窗口之间保留 overlap 个字符的重叠，避免切断上下文。"""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    sentences = [sentence for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            windows.append(current.strip())
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + sentence
        else:
            current += sentence
    if current.strip():
        windows.append(current.strip())
    return windows


def chunk_by_paragraph(
    doc: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """策略二：按小节合并成段落，超长再切窗口。"""
    chunks: list[Chunk] = []
    index = 0
    for section_title, lines in doc.sections:
        merged = " ".join(_strip_item_prefix(line) for line in lines)
        for piece in _split_windows(merged, max_chars, overlap):
            index += 1
            chunks.append(_make_chunk(doc, section_title, "paragraph", index, piece))
    return chunks


def chunk_document(
    doc: Document,
    strategy: str = "rule",
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    if strategy == "rule":
        return chunk_by_rule(doc)
    if strategy == "paragraph":
        return chunk_by_paragraph(doc, max_chars=max_chars, overlap=overlap)
    raise ValueError(f"未知分块策略：{strategy}")


def load_documents(knowledge_dir: Path) -> list[Document]:
    """按文件名排序加载目录下所有 Markdown 文档。"""
    if not knowledge_dir.exists():
        return []
    return [
        parse_document(path)
        for path in sorted(knowledge_dir.glob("*.md"))
        if path.is_file()
    ]


def build_chunks(
    knowledge_dir: Path,
    strategy: str = "rule",
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """对整个知识库目录分块。"""
    chunks: list[Chunk] = []
    for doc in load_documents(knowledge_dir):
        chunks.extend(
            chunk_document(doc, strategy, max_chars=max_chars, overlap=overlap)
        )
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [chunk.char_count for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "doc_count": len({chunk.doc_id for chunk in chunks}),
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_chars": max(lengths, default=0),
        "min_chars": min(lengths, default=0),
    }
