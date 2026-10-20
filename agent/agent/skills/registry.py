"""Skill 包注册表：扫描 `data/skills/*/SKILL.md`，对外只暴露"目录"。

Skills 的核心是**渐进式披露**：

1. 启动时只读 frontmatter（name / description / when_to_use），拼成一份很小的目录；
2. 真正命中某个技能时，才把 `SKILL.md` 正文读进上下文；
3. 正文里引用的 `references/` 再按需加载（下一段做）。

所以这个模块只负责"发现 + 校验 + 取正文"，不负责选技能、也不拼 prompt——
那两件事属于工作流层，分开才能单独测。

校验原则：**坏包只丢自己，不拖垮整张表**。默认把错误收进 `errors` 跳过，
`strict=True` 时才整体报错（CI / `scripts/skill_index.py` 用严格模式）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.config import get_env
from agent.db import DATA_DIR

MAX_SKILL_BYTES = 64 * 1024        # SKILL.md 本体上限
MAX_REFERENCE_BYTES = 256 * 1024   # 单个 references 文件上限
MAX_TOTAL_REFERENCE_BYTES = 512 * 1024  # 一个技能的全部 references 合计上限
MAX_REFERENCES = 5                 # 一个技能最多声明几个细则文件
MAX_BODY_CHARS = 20_000            # 正文注入上限（防一个包吃光上下文）
MIN_DESCRIPTION_CHARS = 8
MAX_DESCRIPTION_CHARS = 300

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
)


class SkillError(ValueError):
    """单个技能包不合法；消息里带路径，方便直接定位。"""


@dataclass(frozen=True)
class Skill:
    """一个已校验的技能包；`body` 不进入常驻目录，只在命中时用。"""

    name: str
    description: str
    when_to_use: str
    version: str
    body: str
    references: tuple[str, ...]
    tags: tuple[str, ...]
    triggers: tuple[str, ...]
    path: Path

    @property
    def directory(self) -> Path:
        return self.path.parent

    def index_entry(self) -> dict:
        """常驻上下文的最小信息：不含正文，避免启动就把所有技能塞进 prompt。"""
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "version": self.version,
            "tags": list(self.tags),
            "triggers": list(self.triggers),
            "references": list(self.references),
            "chars": len(self.body),
            "path": str(self.directory),
        }


class SkillRegistry:
    """一份加载结果：有效技能 + 坏包错误；不抛异常，让调用方决定怎么处理。"""

    def __init__(
        self,
        directory: Path,
        skills: dict[str, Skill] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._skills = dict(skills or {})
        self.errors: list[str] = list(errors or [])

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return sorted(self._skills)

    def skills(self) -> list[Skill]:
        """按名字排序的全部技能（路由用；索引仍走 `index()`）。"""
        return [self._skills[name] for name in self.names()]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(str(name or "").strip())

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            raise SkillError(
                f"没有这个技能：{name!r}；可用技能：{', '.join(self.names()) or '（空）'}"
            )
        return skill

    def index(self) -> list[dict]:
        """按名字排序的目录，直接可以渲染进 system prompt 或接口返回。"""
        return [self._skills[name].index_entry() for name in self.names()]


def default_skills_dir() -> Path:
    """技能根目录；`SKILLS_DIR` 可覆盖（测试/多环境部署用）。"""
    configured = get_env("SKILLS_DIR", "")
    return Path(configured) if configured else DATA_DIR / "skills"


def _text(meta: dict, key: str) -> str:
    value = meta.get(key)
    return str(value).strip() if value is not None else ""


def _string_list(value, field: str, source: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SkillError(f"{source}：frontmatter 的 {field} 必须是字符串列表")
    return tuple(item.strip() for item in value if item.strip())


def _validate_references(skill_dir: Path, entries: tuple[str, ...], source: Path) -> None:
    """引用的文件必须真实存在、且不能跑出技能目录（防 `../` 读任意文件）。"""
    if len(entries) > MAX_REFERENCES:
        raise SkillError(f"{source}：references 最多 {MAX_REFERENCES} 个，收到 {len(entries)} 个")
    root = skill_dir.resolve()
    references_root = (skill_dir / "references").resolve()
    total_bytes = 0
    for entry in entries:
        candidate = Path(entry)
        if candidate.is_absolute():
            raise SkillError(f"{source}：references 不能是绝对路径：{entry}")
        resolved = (skill_dir / candidate).resolve()
        if resolved != root and root not in resolved.parents:
            raise SkillError(f"{source}：references 越界（禁止 ../）：{entry}")
        if resolved != references_root and references_root not in resolved.parents:
            raise SkillError(f"{source}：references 必须放在 references/ 目录下：{entry}")
        if not resolved.is_file():
            raise SkillError(f"{source}：references 文件不存在：{entry}")
        if resolved.stat().st_size > MAX_REFERENCE_BYTES:
            raise SkillError(f"{source}：references 文件超过 {MAX_REFERENCE_BYTES} 字节：{entry}")
        total_bytes += resolved.stat().st_size
    if total_bytes > MAX_TOTAL_REFERENCE_BYTES:
        raise SkillError(
            f"{source}：references 合计超过 {MAX_TOTAL_REFERENCE_BYTES} 字节"
        )


def parse_skill(skill_md: Path) -> Skill:
    """解析并校验一个 `SKILL.md`；任何不合法都抛 `SkillError`。"""
    path = Path(skill_md)
    if not path.is_file():
        raise SkillError(f"{path}：文件不存在")
    if path.stat().st_size > MAX_SKILL_BYTES:
        raise SkillError(f"{path}：超过 {MAX_SKILL_BYTES} 字节")

    raw = path.read_text(encoding="utf-8-sig")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise SkillError(f"{path}：缺少 frontmatter（文件必须以 --- 开头，并以 --- 结束元数据）")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"{path}：frontmatter 不是合法 YAML：{exc}") from exc
    if not isinstance(meta, dict):
        raise SkillError(f"{path}：frontmatter 必须是 key: value 结构")

    name = _text(meta, "name")
    if not NAME_RE.match(name):
        raise SkillError(
            f"{path}：name 必须是小写字母开头的 kebab-case（如 refund-policy），收到 {name!r}"
        )
    if path.parent.name != name:
        raise SkillError(
            f"{path}：目录名 {path.parent.name!r} 必须和 frontmatter 的 name {name!r} 一致"
        )

    description = _text(meta, "description")
    if len(description) < MIN_DESCRIPTION_CHARS:
        raise SkillError(f"{path}：description 至少 {MIN_DESCRIPTION_CHARS} 个字，用来做技能路由")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise SkillError(f"{path}：description 不能超过 {MAX_DESCRIPTION_CHARS} 个字")

    when_to_use = _text(meta, "when_to_use")
    if not when_to_use:
        raise SkillError(f"{path}：缺少 when_to_use（说明什么时候该用这个技能）")

    version = _text(meta, "version") or "0.1.0"
    if not VERSION_RE.match(version):
        raise SkillError(f"{path}：version 必须是 x.y.z，收到 {version!r}")

    body = raw[match.end():].strip()
    if not body:
        raise SkillError(f"{path}：正文为空（frontmatter 之后要有可执行说明）")
    if len(body) > MAX_BODY_CHARS:
        raise SkillError(f"{path}：正文超过 {MAX_BODY_CHARS} 字，应该拆到 references/")

    references = _string_list(meta.get("references"), "references", path)
    _validate_references(path.parent, references, path)

    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        version=version,
        body=body,
        references=references,
        tags=_string_list(meta.get("tags"), "tags", path),
        triggers=_string_list(meta.get("triggers"), "triggers", path),
        path=path,
    )


def load_registry(
    directory: Path | str | None = None, *, strict: bool = False
) -> SkillRegistry:
    """扫描 `directory/*/SKILL.md`；默认跳过坏包并记录原因，`strict` 时整体报错。"""
    root = Path(directory) if directory else default_skills_dir()
    registry = SkillRegistry(root)
    if not root.is_dir():
        registry.errors.append(f"技能目录不存在：{root}")
        if strict:
            raise SkillError("\n".join(registry.errors))
        return registry

    skills: dict[str, Skill] = {}
    for skill_md in sorted(root.glob("*/SKILL.md")):
        try:
            skill = parse_skill(skill_md)
        except SkillError as exc:
            registry.errors.append(str(exc))
            continue
        if skill.name in skills:
            registry.errors.append(
                f"{skill.path}：技能名重复：{skill.name}（已有 {skills[skill.name].path}）"
            )
            continue
        skills[skill.name] = skill

    registry = SkillRegistry(root, skills, registry.errors)
    if strict and registry.errors:
        raise SkillError("\n".join(registry.errors))
    return registry


_REGISTRY: SkillRegistry | None = None


def get_registry(*, refresh: bool = False) -> SkillRegistry:
    """进程内缓存：技能包是静态文件，启动加载一次即可；测试/热更新用 refresh=True。"""
    global _REGISTRY
    if refresh or _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
