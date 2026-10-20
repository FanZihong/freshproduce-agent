"""分块器测试：文档解析、两套策略差异、chunk_id 稳定性。"""

from __future__ import annotations

from agent.db import DATA_DIR
from agent.rag.chunker import build_chunks, chunk_stats, load_documents

KNOWLEDGE_DIR = DATA_DIR / "knowledge"


def test_loads_five_documents_with_source_and_sections():
    documents = load_documents(KNOWLEDGE_DIR)
    assert len(documents) == 5
    assert all(document.source for document in documents)
    assert all(document.sections for document in documents)


def test_rule_chunks_have_required_fields():
    chunks = build_chunks(KNOWLEDGE_DIR, strategy="rule")
    assert chunks
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.doc_id
        assert chunk.section
        assert chunk.text
        assert chunk.char_count == len(chunk.text)
        assert chunk.source


def test_chunk_ids_are_unique_and_deterministic():
    first = build_chunks(KNOWLEDGE_DIR, strategy="rule")
    second = build_chunks(KNOWLEDGE_DIR, strategy="rule")
    ids = [chunk.chunk_id for chunk in first]
    assert len(ids) == len(set(ids))
    assert ids == [chunk.chunk_id for chunk in second]


def test_two_strategies_produce_different_granularity():
    rule_stats = chunk_stats(build_chunks(KNOWLEDGE_DIR, strategy="rule"))
    paragraph_stats = chunk_stats(build_chunks(KNOWLEDGE_DIR, strategy="paragraph"))
    assert rule_stats["chunk_count"] > paragraph_stats["chunk_count"]
    assert paragraph_stats["avg_chars"] > rule_stats["avg_chars"]


def test_paragraph_windows_respect_max_chars():
    chunks = build_chunks(KNOWLEDGE_DIR, strategy="paragraph", max_chars=120, overlap=20)
    # 允许因重叠略超窗口，但不应出现远超窗口的 chunk
    assert all(chunk.char_count <= 200 for chunk in chunks)
