"""Core utilities and registry for style packs."""
from __future__ import annotations

from collections import defaultdict
from importlib import import_module
import pkgutil
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, HttpUrl

StyleKind = Literal[
    "conversation_brevity",
    "conversation_scene",
    "author",
    "output_rails",
]


class StyleMeta(BaseModel):
    id: str
    name: str
    description: str
    avatar_url: HttpUrl | None = None
    version: str = "1.0.0"


class StyleConstraints(BaseModel):
    forbidden_endings: list[str] = Field(default_factory=list)
    emoji_quota: int = 1


class StyleTemplates(BaseModel):
    conversation_brevity: str
    conversation_scene: str
    author: str
    output_rails: str


class StylePack(BaseModel):
    meta: StyleMeta
    constraints: StyleConstraints
    templates: StyleTemplates
    initial_situation: str | None = None


class _SafeDict(defaultdict):
    def __missing__(self, key):  # type: ignore[override]
        return ""


def render_style_template(template: str, ctx: Dict[str, Any]) -> str:
    return template.format_map(_SafeDict(str, **ctx))


STYLE_PACKS: Dict[str, StylePack] = {}


def register_style(pack: StylePack) -> None:
    STYLE_PACKS[pack.meta.id] = pack


def get_style_pack(style_id: str) -> StylePack:
    if style_id in STYLE_PACKS:
        return STYLE_PACKS[style_id]
    return STYLE_PACKS["henry"]


def list_style_meta() -> list[StyleMeta]:
    return [p.meta for p in STYLE_PACKS.values()]


_loaded_builtin_styles = False


def load_builtin_styles() -> None:
    global _loaded_builtin_styles
    if _loaded_builtin_styles:
        return

    package_name = f"{__package__}.definitions"
    package = import_module(package_name)

    for module_info in pkgutil.iter_modules(package.__path__):  # type: ignore[attr-defined]
        if module_info.name.startswith("_"):
            continue
        import_module(f"{package_name}.{module_info.name}")

    _loaded_builtin_styles = True


__all__ = [
    "StyleKind",
    "StyleMeta",
    "StyleConstraints",
    "StyleTemplates",
    "StylePack",
    "render_style_template",
    "STYLE_PACKS",
    "register_style",
    "get_style_pack",
    "list_style_meta",
    "load_builtin_styles",
]
