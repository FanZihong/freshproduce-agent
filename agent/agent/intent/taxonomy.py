"""意图体系：加载 taxonomy.yaml、校验，并提供「标签 -> 路由分支」的统一映射。

设计要点：意图类别（10 类）与图上的路由分支（5 个节点）不是一一对应，
映射关系集中在这里，避免 router 里再散落一份硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = AGENT_ROOT / "data" / "intent" / "taxonomy.yaml"

VALID_ROUTES: frozenset[str] = frozenset({"order", "claim", "faq", "handoff", "fallback"})

# 代码里沿用的旧标签 -> 体系标签，保证路由与评测能对上
ALIASES: dict[str, str] = {
    "knowledge_qa": "quality_qa",
    "fallback": "chitchat",
}


@dataclass(frozen=True)
class Confusion:
    with_label: str
    tip: str


@dataclass(frozen=True)
class IntentSpec:
    key: str
    name: str
    route: str
    definition: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    confusions: tuple[Confusion, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: int
    updated: str
    description: str
    labels: tuple[IntentSpec, ...]
    routing_priority: tuple[str, ...]
    multi_intent_policy: dict
    out_of_domain: dict
    labeling_guidelines: tuple[str, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.labels)

    def spec(self, key: str) -> IntentSpec | None:
        resolved = ALIASES.get(key, key)
        for item in self.labels:
            if item.key == resolved:
                return item
        return None

    def route_of(self, key: str) -> str:
        """标签 -> 路由分支；未知标签一律走兜底分支。"""
        spec = self.spec(key)
        return spec.route if spec else "fallback"

    def name_of(self, key: str) -> str:
        spec = self.spec(key)
        return spec.name if spec else key


@lru_cache(maxsize=4)
def _load(path_str: str) -> Taxonomy:
    payload = yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))
    labels = []
    for item in payload["labels"]:
        labels.append(
            IntentSpec(
                key=item["key"],
                name=item["name"],
                route=item["route"],
                definition=item["definition"],
                includes=tuple(item.get("includes", [])),
                excludes=tuple(item.get("excludes", [])),
                positives=tuple(item.get("positives", [])),
                negatives=tuple(item.get("negatives", [])),
                confusions=tuple(
                    Confusion(with_label=conf["with"], tip=conf["tip"])
                    for conf in item.get("confusions", [])
                ),
            )
        )
    return Taxonomy(
        version=int(payload.get("version", 1)),
        updated=str(payload.get("updated", "")),
        description=str(payload.get("description", "")),
        labels=tuple(labels),
        routing_priority=tuple(payload.get("routing_priority", [])),
        multi_intent_policy=dict(payload.get("multi_intent_policy", {})),
        out_of_domain=dict(payload.get("out_of_domain", {})),
        labeling_guidelines=tuple(payload.get("labeling_guidelines", [])),
    )


def load_taxonomy(path: Path | str = TAXONOMY_PATH) -> Taxonomy:
    return _load(str(path))


def validate(taxonomy: Taxonomy | None = None, *, min_labels: int = 10) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    taxonomy = taxonomy or load_taxonomy()
    problems: list[str] = []

    if len(taxonomy.labels) < min_labels:
        problems.append(f"意图数量不足：{len(taxonomy.labels)} < {min_labels}")

    keys = taxonomy.keys
    if len(set(keys)) != len(keys):
        problems.append("存在重复的意图 key")

    for spec in taxonomy.labels:
        prefix = f"[{spec.key}]"
        if not spec.name or not spec.definition:
            problems.append(f"{prefix} 缺少 name 或 definition")
        if spec.route not in VALID_ROUTES:
            problems.append(f"{prefix} route 非法：{spec.route}")
        if len(spec.positives) < 3:
            problems.append(f"{prefix} 正例少于 3 条")
        if len(spec.negatives) < 2:
            problems.append(f"{prefix} 反例少于 2 条")
        for confusion in spec.confusions:
            if confusion.with_label not in keys:
                problems.append(f"{prefix} 易混提示指向了不存在的类别：{confusion.with_label}")

    missing_in_priority = set(keys) - set(taxonomy.routing_priority)
    if missing_in_priority:
        problems.append(f"routing_priority 未覆盖：{sorted(missing_in_priority)}")

    if not taxonomy.out_of_domain:
        problems.append("缺少 out_of_domain 配置")

    return problems


def route_of(label: str) -> str:
    """便捷函数：标签（含代码旧标签）-> 路由分支。"""
    return load_taxonomy().route_of(label)


def label_name(label: str) -> str:
    return load_taxonomy().name_of(label)
