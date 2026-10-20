"""FAQ / 政策问答工具：检索 + 引用 + "无据不作答"。

有据原则：
1. 答案只使用检索到的 chunk 原文，不做自由发挥；
2. 每条结论都带引用（文档名 / 小节 / 来源）；
3. 查询里出现**知识库不认识的实体词**（词表里没有、且至少一个字在知识库里从未出现过），
   或最典型的关键词不在首条命中里、覆盖率不足 → 判定无据，由上层节点如实告知并转人工，
   而不是硬编一个答案。

为什么"整词不在词表"不能直接判无据（2026-10-15 端到端评测暴露）：
分词是字符 bigram，"退款一般几天到账" 会被切成「几天」「天到」「款一」「般几」这类
跨词碎片，它们的 df=0 只是分词假象；旧版挑"最稀有的词"当锚点，经常挑中这种碎片，
于是知识库里明明有答案（法规/FAQ 里写着"退款 1 至 3 个工作日到账"）也判无据、误转人工。
现在改成两步：
① **未知实体否决**：整词不在词表里，且"整词每个字都没在知识库出现过"（山竹：山、竹都没出现）
   或"它是句子的第一个内容词且带陌生字"（椰子：子常见、椰陌生，但它出现在主语位置）；
② **已知词覆盖率**：只拿知识库认识的词当分母，覆盖率过半才算有据。
实测（离线复现，见 eval/reports 的说明）：qa_golden 的 25 条 direct 问句从 1 条判有据
提升到 19 条，同时山竹/椰子/荔枝/释迦这些真未知实体仍然全部判无据。
"""

from __future__ import annotations

from agent.config import get_env
from agent.rag.hybrid_search import tokenize
from agent.rag.retriever import get_retriever

DEFAULT_TOP_K = 3
MIN_COVERAGE = 0.5

# 疑问词与助词，参与区分度计算时忽略
STOPWORDS: frozenset[str] = frozenset(
    {
        "怎么", "么保", "什么", "可以", "吗", "呢", "的", "了", "是", "有", "能", "要",
        "该", "如何", "多少", "样子", "处理", "情况", "问题", "请问", "一下", "这个", "那个",
        "为什么", "多久", "哪些", "哪种", "还是", "或者",
    }
)

# 问题类型词 / 连接词：用户是在问"哪一类"，不是在指一个知识库必须认识的实体。
# 「几天」这类词整字都不在知识库里（几=0 次），但不能据此判无据。
NON_ENTITY_WORDS: frozenset[str] = frozenset(
    {
        "几天", "几号", "几次", "几年", "几个月", "多少天", "多长时间", "多久", "多久能",
        "一般", "大概", "大约", "左右", "另外", "还有", "顺便", "同时", "以及", "是否",
    }
)

_DF_CACHE: dict[tuple[str, int], dict[str, int]] = {}
_CHAR_CACHE: dict[tuple[str, int], set[str]] = {}


def _document_frequency(retriever) -> dict[str, int]:
    """统计词在整个知识库里出现在多少个 chunk 中，用于挑"最具区分度的词"。"""
    key = (getattr(retriever, "strategy", "rule"), len(retriever.chunks))
    if key not in _DF_CACHE:
        counter: dict[str, int] = {}
        for chunk in retriever.chunks:
            for token in set(tokenize(chunk.text)):
                counter[token] = counter.get(token, 0) + 1
        _DF_CACHE[key] = counter
    return _DF_CACHE[key]


def _known_characters(retriever) -> set[str]:
    """知识库里出现过的所有单字：用来区分"真未知实体"和"分词碎片"。"""
    key = (getattr(retriever, "strategy", "rule"), len(retriever.chunks))
    if key not in _CHAR_CACHE:
        characters: set[str] = set()
        for chunk in retriever.chunks:
            characters |= set(chunk.text)
        _CHAR_CACHE[key] = characters
    return _CHAR_CACHE[key]


def _content_tokens(text: str) -> set[str]:
    """参与"有据"判断的内容词：去停用词、去掉单字（单字区分度太低）。"""
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) >= 2}


def _unknown_entity(query: str, df: dict[str, int], known_chars: set[str]) -> str | None:
    """找出"知识库不认识的实体词"；没有就返回 None。

    只看"不在词表里（df=0）"的词，命中下面任一条即算：
    ① 整词**每个字**都没在知识库出现过 —— 一定是知识库不认识的东西（山竹、荔枝、释迦）；
    ② 它是句子里的**第一个内容词**且带陌生的字 —— 主语位置上的陌生词（椰子：子常见、椰陌生）。

    为什么要有"单字"这层：bigram 会切出跨词碎片，「可以无理由」切出「以无」「鲜可」，
    它们的字都在知识库里出现过，属于分词假象；只有整词的字全陌生，才是真新词。
    为什么要限定"第一个内容词"：碎片大多出现在句子的中后段（「般几」「天到」），
    而实体的位置通常在句首（"椰子怎么保存"）。
    """
    ordered = [token for token in tokenize(query) if token not in STOPWORDS and len(token) >= 2]
    head = next((token for token in ordered if token not in NON_ENTITY_WORDS), None)
    candidates = sorted(
        {token for token in ordered if token not in NON_ENTITY_WORDS and df.get(token, 0) == 0}
    )
    for token in candidates:
        if all(char not in known_chars for char in token):
            return token
        if token == head and any(char not in known_chars for char in token):
            return token
    return None


def _anchor_token(query: str, df: dict[str, int]) -> tuple[str | None, int]:
    """挑出查询里最有区分度的**已知**内容词（在知识库中出现次数最少）。

    词表外的词不参选：真未知实体已经被 _unknown_entity 拦下，剩下的词表外词都是分词碎片，
    拿碎片当锚点只会误伤（旧版就是被「以无」「几天」这类碎片坑的）。
    """
    known = [token for token in _content_tokens(query) if df.get(token, 0) > 0]
    if not known:
        return None, 0
    anchor = min(known, key=lambda token: (df[token], -len(token), token))
    return anchor, df[anchor]


def _coverage(query: str, text: str, df: dict[str, int]) -> float:
    """**已知**内容词在候选文本里的覆盖率，用于门槛判断。

    分母只算知识库认识的词：要求候选文本包含一个知识库里从没出现过的词是没有意义的。
    """
    tokens = {token for token in _content_tokens(query) if df.get(token, 0) > 0}
    if not tokens:
        return 0.0
    return len(tokens & set(tokenize(text))) / len(tokens)


def build_citations(query: str, hits: list[dict]) -> list[dict]:
    """按"有据"规则从候选里挑引用。

    规则：① 同文档同小节只保留排名最高的一条；② 与问题没有内容词交集的候选不作为依据。
    """
    query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, hit in enumerate(hits):
        key = (hit["doc_title"], hit["section"])
        if key in seen:
            continue
        if index > 0 and not (query_tokens & set(tokenize(hit["text"]))):
            continue
        seen.add(key)
        citations.append(
            {
                "chunk_id": hit["chunk_id"],
                "title": hit["doc_title"],
                "section": hit["section"],
                "source": hit["source"],
                "snippet": hit["text"][:80],
                "score": hit["rerank_score"],
            }
        )
    return citations


def search_faq(query: str, context: dict | None = None, top_k: int = DEFAULT_TOP_K) -> dict:
    """检索知识库并返回带引用的结果；无据时 found=False。"""
    reranker_kind = get_env("FAQ_RERANK", "score") or "score"
    # 检索路：默认混合（向量 + BM25 + RRF）。`RETRIEVAL_KIND=vector` 可切纯向量——
    # 消融证据（eval/reports/retrieval_findings.md 与 rerank_iteration.md）显示纯向量在这份
    # 30 条问句上更强（recall@5 0.90 vs 0.8667），但混合路当初是为了字面查询（订单号、生僻词）
    # 的鲁棒性加进来的，而且切它属于"改链路"，必须先过回归（对话端 + judge）再决定。
    retrieval_kind = get_env("RETRIEVAL_KIND", "hybrid") or "hybrid"
    retriever = get_retriever(reranker_kind=reranker_kind)
    debug = retriever.search_debug(query, context, top_k, kind=retrieval_kind)

    hits = debug["reranked"]
    if not hits:
        return {
            "query": query,
            "rewritten": debug["rewrite"]["query"],
            "found": False,
            "reason": "知识库没有返回任何候选",
            "anchor": None,
            "anchor_df": 0,
            "top_score": 0.0,
            "hits": [],
            "citations": [],
        }

    top = hits[0]
    rewritten = debug["rewrite"]["query"]
    df = _document_frequency(retriever)
    known_chars = _known_characters(retriever)
    unknown = _unknown_entity(rewritten, df, known_chars)
    anchor, anchor_df = _anchor_token(rewritten, df)
    known_tokens = sorted(token for token in _content_tokens(rewritten) if df.get(token, 0) > 0)
    coverage = _coverage(rewritten, top["text"], df)
    # 真正在候选里出现的已知词：写进 reason，方便排查"凭什么说有据"。
    # 按出现频次从高到低排（高频词更像"核心词"，稀有组合多半是分词碎片），只留 3 个。
    hit_tokens = [token for token in known_tokens if token in set(tokenize(top["text"]))]
    support = sorted(
        [token for token in hit_tokens if token not in NON_ENTITY_WORDS] or hit_tokens,
        key=lambda token: (-df[token], -len(token), token),
    )[:3]

    if unknown is not None:
        # 真未知实体（山竹/椰子）：知识库里没有这个东西，不能拿别的商品的条目硬答
        found = False
        reason = f"知识库中没有「{unknown}」相关内容"
    elif anchor is None:
        # 查询里的词一个都不在知识库词表内：没有任何可核对的依据
        found = False
        reason = "查询里的词都不在知识库词表内，无法核对依据"
    elif coverage < MIN_COVERAGE:
        # 覆盖率是唯一的内容门槛：过半已知词都命中候选，才算"有依据"。
        # 不再单独要求"最稀有的那个词出现在首条里"——那是覆盖率已经覆盖的约束，
        # 而且更严：它把「成赔」「工审」这类低频组合词变成硬门槛，整句就废了。
        found = False
        reason = f"关键词覆盖率仅 {coverage:.2f}，依据不足"
    else:
        found = True
        reason = (
            f"命中「{'」「'.join(support)}」等 {len(known_tokens)} 个关键词，"
            f"覆盖率 {coverage:.2f}"
        )

    citations = build_citations(debug["rewrite"]["query"], hits)
    return {
        "query": query,
        "rewritten": debug["rewrite"]["query"],
        "found": found,
        "reason": reason,
        "anchor": anchor,
        "anchor_df": anchor_df,
        "coverage": round(coverage, 4),
        "known_tokens": known_tokens,
        "support_terms": support,
        "unknown_entity": unknown,
        "top_score": float(top["rerank_score"]),
        "reranker": debug["reranker"],
        "hits": hits,
        "citations": citations,
    }
