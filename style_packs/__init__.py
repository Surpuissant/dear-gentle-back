"""Style pack registry and built-in style loading."""
from .core import (
    StyleKind,
    StyleMeta,
    StyleConstraints,
    StyleTemplates,
    StylePack,
    render_style_template,
    STYLE_PACKS,
    register_style,
    get_style_pack,
    list_style_meta,
    load_builtin_styles,
)

load_builtin_styles()

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
