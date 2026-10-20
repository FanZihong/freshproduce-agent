"""search_faq 工具与知识问答分支测试。"""

from __future__ import annotations

import pytest

from agent.graph.workflow import build_graph
from agent.rag.embedder import HashingEmbedder
from agent.rag.retriever import Retriever
from agent.tools import gateway
from agent.tools.degradation import ASK_AGAIN, HANDOFF
from agent.tools.faq import search_faq
from agent.tools.gateway import ToolSpec, call_tool
from agent.tools.schemas import SearchFaqInput


@pytest.fixture
def fast_retriever(monkeypatch):
    """用哈希向量替换真实模型，保证测试不依赖模型下载。"""
    retriever = Retriever(embedder=HashingEmbedder())
    monkeypatch.setattr("agent.tools.faq.get_retriever", lambda **kwargs: retriever)
    return retriever


def test_search_faq_returns_citations(fast_retriever):
    result = search_faq("车厘子怎么保存")
    assert result["found"] is True
    assert result["citations"]
    citation = result["citations"][0]
    assert citation["title"]
    assert citation["section"]
    assert citation["source"]
    assert "车厘子" in result["hits"][0]["text"]


def test_search_faq_reports_no_evidence_for_unknown_entity(fast_retriever):
    result = search_faq("山竹怎么保存")
    assert result["found"] is False
    assert "山竹" in result["reason"]


def test_gateway_rejects_empty_query(fast_retriever):
    result = call_tool("search_faq", query="")
    assert result.ok is False
    assert result.kind == "validation"
    assert result.action == ASK_AGAIN


def test_gateway_degrades_to_handoff_on_error(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("知识库不可用")

    monkeypatch.setitem(
        gateway.TOOL_REGISTRY,
        "search_faq",
        ToolSpec(SearchFaqInput, broken, attempts=1, base_delay=0.01),
    )
    result = call_tool("search_faq", query="车厘子怎么保存")
    assert result.ok is False
    assert result.action == HANDOFF
    assert "知识库" in (result.reply or "")


def test_faq_node_answers_with_citations(fast_retriever):
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-faq"}}
    graph.invoke({"user_input": "车厘子怎么保存"}, config=config)

    state = graph.get_state(config).values
    assert state["handoff"] is False
    assert "参考来源" in state["reply"]
    assert state["citations"]
    # 状态里存的是**体系里的规范标签**：规则版的 knowledge_qa 是历史别名，
    # 经 taxonomy.ALIASES 归一后写进 state（路由仍然走 faq 分支，见 route_of）
    assert state["intent"] == "quality_qa"
    # 注意：经过 checkpoint 的 JSON 序列化后，元组会变成列表，所以这里用 list() 比较
    assert list(state["intent_labels"]) == ["quality_qa"]


def test_faq_node_hands_off_when_no_evidence(fast_retriever):
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-faq-none"}}
    graph.invoke({"user_input": "山竹怎么保存"}, config=config)

    state = graph.get_state(config).values
    assert state["handoff"] is True
    assert "没有找到可靠的依据" in state["reply"]
    assert "知识库无据" in state["handoff_summary"]


# ---------------- 有据门槛：分词碎片不能当"未知实体" ----------------
#
# 端到端评测（dg-14 / dg-15）暴露的坑：分词是字符 bigram，
# 「退款一般几天到账」切出「几天」「般几」，「生鲜可以无理由退货吗」切出「以无」「鲜可」。
# 旧版拿"最稀有的词"当锚点，正好挑中这些 df=0 的碎片，于是知识库里明明写着
# 「退款 1 至 3 个工作日到账」「生鲜不支持无理由退货」，系统却判无据、误转人工。

FAKE_KB: list[str] = [
    "问：车厘子怎么保存？答：冷藏 0 至 4 度，食用前再清洗，建议 3 天内吃完。",
    "问：生鲜可以无理由退货吗？答：生鲜商品因保质期短，原则上不支持无理由退货，质量问题按赔付处理而非退货。",
    "问：退款多久到账？答：审核通过后一般 1 至 3 个工作日原路退回。",
]


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFaqRetriever:
    """只实现 search_faq 需要的接口：chunks（算词频用）+ search_debug（给候选）。"""

    strategy = "fake"

    def __init__(self, chunks: list[str], top: str) -> None:
        self.chunks = [_FakeChunk(text) for text in chunks]
        self._top = top
        self.calls: list[dict] = []

    def search_debug(self, query, context=None, top_k=3, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return {
            "rewrite": {"query": query},
            "reranked": [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "doc_title": "常见问题FAQ",
                    "section": "退款与发票",
                    "text": self._top,
                    "source": "data/knowledge/常见问题FAQ.md",
                    "rerank_score": 1.0,
                }
            ],
            "reranker": "score",
        }


def _ask_with_fake_kb(query: str, top: str, monkeypatch) -> dict:
    retriever = _FakeFaqRetriever(FAKE_KB, top)
    monkeypatch.setattr("agent.tools.faq.get_retriever", lambda **kwargs: retriever)
    return search_faq(query)


def _patch_faq_env(monkeypatch, **values) -> None:
    """把 faq 模块读环境变量的入口换成受控字典。

    不能靠 `monkeypatch.delenv` 来测"默认值"：`get_env` 会顺带加载 `agent/.env`，
    本机 .env 里一旦写了 `RETRIEVAL_KIND=vector`，删掉进程环境变量也没用——
    配置会从文件里被读回来。单测必须与本机配置无关。
    """

    def fake_get_env(key: str, default: str = "") -> str:
        return values.get(key, default)

    monkeypatch.setattr("agent.tools.faq.get_env", fake_get_env)


def test_search_faq_answers_paraphrase_covered_by_kb(monkeypatch):
    """「几天」是问时间，不是在指一个知识库必须认识的实体。"""
    result = _ask_with_fake_kb("退款一般几天到账", FAKE_KB[2], monkeypatch)
    assert result["found"] is True
    assert result["citations"]
    assert result["unknown_entity"] is None


def test_search_faq_ignores_bigram_fragments(monkeypatch):
    """「以无」「鲜可」每个字都在知识库里出现过 —— 是分词假象，不是新实体。"""
    result = _ask_with_fake_kb("生鲜可以无理由退货吗", FAKE_KB[1], monkeypatch)
    assert result["found"] is True
    assert result["citations"]
    assert result["unknown_entity"] is None


def test_search_faq_still_rejects_absent_product(monkeypatch):
    """安全属性不能丢：知识库里没这个商品就如实说没有。"""
    result = _ask_with_fake_kb("山竹怎么保存", FAKE_KB[0], monkeypatch)
    assert result["found"] is False
    assert result["unknown_entity"] == "山竹"
    assert "山竹" in result["reason"]


def test_unknown_entity_only_flags_foreign_words():
    from agent.tools.faq import _document_frequency, _known_characters, _unknown_entity

    retriever = _FakeFaqRetriever(FAKE_KB, FAKE_KB[0])
    df = _document_frequency(retriever)
    characters = _known_characters(retriever)

    assert _unknown_entity("山竹怎么保存", df, characters) == "山竹"
    assert _unknown_entity("椰子怎么保存", df, characters) == "椰子"   # 位于主语位置的陌生词
    assert _unknown_entity("退款一般几天到账", df, characters) is None
    assert _unknown_entity("生鲜可以无理由退货吗", df, characters) is None


# ---------------- 检索路开关：默认混合，可切纯向量 ----------------


def test_search_faq_uses_hybrid_by_default_and_vector_when_switched(monkeypatch):
    from agent.tools.faq import search_faq

    retriever = _FakeFaqRetriever(FAKE_KB, FAKE_KB[0])
    monkeypatch.setattr("agent.tools.faq.get_retriever", lambda **kwargs: retriever)

    _patch_faq_env(monkeypatch, FAQ_RERANK="score")
    search_faq("车厘子怎么保存")
    assert retriever.calls[-1]["kind"] == "hybrid"          # 默认不改变现有行为

    _patch_faq_env(monkeypatch, FAQ_RERANK="score", RETRIEVAL_KIND="vector")
    search_faq("车厘子怎么保存")
    assert retriever.calls[-1]["kind"] == "vector"          # 环境变量可切
