"""知识库接口：真实读取知识库目录并统计分块。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter

from agent.rag.chunker import chunk_by_paragraph, chunk_by_rule, load_documents

from app.api.schemas import KnowledgeDocOut, KnowledgeDocRequest
from app.config import AGENT_ROOT

router = APIRouter(tags=["knowledge"])

KNOWLEDGE_DIR = AGENT_ROOT / "data" / "knowledge"

_EXTRA_DOCS: list[dict] = []

_CATEGORY_KEYWORDS = {
    "赔付规则": "赔付规则",
    "售后": "售后政策",
    "政策": "售后政策",
    "faq": "FAQ",
    "物流": "物流",
    "储存": "商品知识",
}


def _guess_category(name: str) -> str:
    lowered = name.lower()
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return category
    return "售后政策"


def _scan_docs() -> list[dict]:
    """扫描知识库目录，按真实分块结果统计每个文档的 chunk 数。"""
    docs: list[dict] = []
    for index, document in enumerate(load_documents(KNOWLEDGE_DIR)):
        path = Path(document.path)
        rule_chunks = chunk_by_rule(document)
        paragraph_chunks = chunk_by_paragraph(document)
        docs.append(
            {
                "id": f"doc-{index:02d}",
                "docId": document.doc_id,
                "name": path.name,
                "category": _guess_category(f"{path.name} {document.title}"),
                "chunkCount": len(rule_chunks),
                "chunkCountParagraph": len(paragraph_chunks),
                "sections": len(document.sections),
                "source": document.source,
                "status": "indexed",
                "updatedAt": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    return docs + _EXTRA_DOCS


@router.get("/knowledge/docs", response_model=list[KnowledgeDocOut])
def list_docs() -> list[dict]:
    return _scan_docs()


@router.post("/knowledge/docs", response_model=KnowledgeDocOut)
def create_doc(payload: KnowledgeDocRequest) -> dict:
    doc = {
        "id": f"doc-{uuid4().hex[:6]}",
        "name": payload.fileName,
        "category": payload.category,
        "chunkCount": 0,
        "status": "indexing",
        "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _EXTRA_DOCS.insert(0, doc)
    return doc
