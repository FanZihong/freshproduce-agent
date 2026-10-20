"""护栏规则注册表：扫描 `data/rails/*.yaml`，坏规则只丢自己。"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from agent.config import get_env
from agent.db import DATA_DIR
from agent.rails.models import ACTIONS, EVENT_TYPES, SCOPES, RailRule
from agent.rails.matcher import OPS

RULE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{1,63}$")


class RailConfigError(ValueError):
    """单条规则或单个文件不合法。"""


class RailRegistry:
    def __init__(
        self,
        directory: Path,
        rules: dict[str, RailRule] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._rules = dict(rules or {})
        self.errors = list(errors or [])

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def ids(self) -> list[str]:
        return sorted(self._rules)

    def get(self, rule_id: str) -> RailRule | None:
        return self._rules.get(str(rule_id or "").strip())

    def rules(self, event: str | None = None, scope: str | None = None) -> list[RailRule]:
        items = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if event and rule.event not in {event, "*"}:
                continue
            if scope and rule.scope != scope:
                continue
            items.append(rule)
        return sorted(items, key=lambda rule: (-rule.priority, rule.id))

    def index(self) -> list[dict]:
        return [
            {
                "id": rule.id,
                "scope": rule.scope,
                "event": rule.event,
                "action": rule.action,
                "priority": rule.priority,
                "reason": rule.reason,
                "keywords": list(rule.keywords),
                "regex": rule.regex,
                "not_regex": rule.not_regex,
                "conditions": list(rule.conditions),
                "messages": rule.message,
                "enabled": rule.enabled,
                "source": rule.source,
            }
            for rule in (self._rules[name] for name in self.ids())
        ]


def default_rails_dir() -> Path:
    configured = get_env("RAILS_DIR", "")
    return Path(configured) if configured else DATA_DIR / "rails"


def _parse_rule(raw: dict, source: str) -> RailRule:
    if not isinstance(raw, dict):
        raise RailConfigError(f"{source}：规则必须是 key: value 结构")
    rule_id = str(raw.get("id") or "").strip()
    if not RULE_ID_RE.match(rule_id):
        raise RailConfigError(f"{source}：规则 id 必须是 kebab-case/点号，收到 {rule_id!r}")

    scope = str(raw.get("scope") or "").strip()
    if scope not in SCOPES:
        raise RailConfigError(f"{source}：scope 必须是 {SCOPES}，收到 {scope!r}")

    action = str(raw.get("action") or "").strip()
    if action not in ACTIONS:
        raise RailConfigError(f"{source}：action 必须是 {ACTIONS}，收到 {action!r}")

    match = raw.get("match") or {}
    if not isinstance(match, dict):
        raise RailConfigError(f"{source}：match 必须是 key: value 结构")
    event = str(match.get("event") or "").strip()
    if event not in EVENT_TYPES and event != "*":
        raise RailConfigError(f"{source}：match.event 必须是 {sorted(EVENT_TYPES)} 或 *，收到 {event!r}")

    keywords = match.get("keywords") or []
    if not isinstance(keywords, list) or any(not isinstance(item, str) for item in keywords):
        raise RailConfigError(f"{source}：match.keywords 必须是字符串列表")
    keywords = [item.strip() for item in keywords if item.strip()]

    regex = str(match.get("regex") or "").strip()
    if regex:
        try:
            re.compile(regex)
        except re.error as exc:
            raise RailConfigError(f"{source}：match.regex 不合法：{exc}") from exc

    not_regex = str(match.get("not_regex") or "").strip()
    if not_regex:
        try:
            re.compile(not_regex)
        except re.error as exc:
            raise RailConfigError(f"{source}：match.not_regex 不合法：{exc}") from exc

    conditions = match.get("conditions") or []
    if not isinstance(conditions, list):
        raise RailConfigError(f"{source}：match.conditions 必须是列表")
    for condition in conditions:
        if not isinstance(condition, dict) or not condition.get("field"):
            raise RailConfigError(f"{source}：每个 condition 必须有 field")
        if str(condition.get("op") or "==") not in OPS:
            raise RailConfigError(f"{source}：condition.op 必须是 {OPS}")

    if not keywords and not regex and not not_regex and not conditions:
        raise RailConfigError(f"{source}：规则至少要有关键词、正则、反向正则或条件之一")

    message = str(raw.get("message") or "").strip()
    if action in {"block", "handoff", "rewrite"} and not message:
        raise RailConfigError(f"{source}：{action} 动作必须提供 message")

    priority = raw.get("priority", 50)
    try:
        priority = int(priority)
    except (TypeError, ValueError) as exc:
        raise RailConfigError(f"{source}：priority 必须是整数") from exc

    min_score = match.get("min_score", 3.0)
    try:
        min_score = float(min_score)
    except (TypeError, ValueError) as exc:
        raise RailConfigError(f"{source}：match.min_score 必须是数字") from exc

    tags = raw.get("tags") or []
    if not isinstance(tags, list) or any(not isinstance(item, str) for item in tags):
        raise RailConfigError(f"{source}：tags 必须是字符串列表")

    return RailRule(
        id=rule_id,
        scope=scope,
        event=event,
        priority=priority,
        action=action,
        message=message,
        reason=str(raw.get("reason") or "").strip(),
        keywords=tuple(keywords),
        regex=regex,
        not_regex=not_regex,
        conditions=tuple(dict(item) for item in conditions),
        tags=tuple(item.strip() for item in tags if item.strip()),
        min_score=min_score,
        enabled=bool(raw.get("enabled", True)),
        source=source,
    )


def load_registry(directory: Path | str | None = None, *, strict: bool = False) -> RailRegistry:
    root = Path(directory) if directory else default_rails_dir()
    registry = RailRegistry(root)
    if not root.is_dir():
        registry.errors.append(f"护栏目录不存在：{root}")
        if strict:
            raise RailConfigError("\n".join(registry.errors))
        return registry

    rules: dict[str, RailRule] = {}
    for path in sorted(root.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        except yaml.YAMLError as exc:
            registry.errors.append(f"{path}：YAML 解析失败：{exc}")
            continue
        raw_rules = payload.get("rules") if isinstance(payload, dict) else payload
        if not isinstance(raw_rules, list):
            registry.errors.append(f"{path}：文件必须是规则列表，或 {{rules: [...]}}")
            continue
        for position, raw in enumerate(raw_rules):
            try:
                rule = _parse_rule(raw, f"{path.name}#{position + 1}")
            except RailConfigError as exc:
                registry.errors.append(str(exc))
                continue
            if rule.id in rules:
                registry.errors.append(f"{path.name}：规则 id 重复：{rule.id}")
                continue
            rules[rule.id] = rule

    registry = RailRegistry(root, rules, registry.errors)
    if strict and registry.errors:
        raise RailConfigError("\n".join(registry.errors))
    return registry


_REGISTRY: RailRegistry | None = None


def get_registry(*, refresh: bool = False) -> RailRegistry:
    global _REGISTRY
    if refresh or _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
