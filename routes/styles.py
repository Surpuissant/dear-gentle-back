# routes/styles.py
# FastAPI router for style pack endpoints and lightweight in-memory editor.

from pathlib import Path

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from style_packs import (
    STYLE_PACKS,
    StylePack,
    get_style_pack,
    list_style_meta,
    register_style,
)
from stores import USERS

router = APIRouter(prefix="/api/styles", tags=["styles"])
lab_router = APIRouter(tags=["styles-lab"])

LAB_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "style_lab.html"
)


class SelectStyleBody(BaseModel):
    user_id: str
    style_id: str


@router.get("")
def get_styles():
    # Return list of style meta with initial situation details
    items = []
    for meta in list_style_meta():
        pack = get_style_pack(meta.id)
        item = meta.model_dump()
        item["initial_situation"] = pack.initial_situation
        items.append(item)
    return {"ok": True, "items": items}


@router.get("/{style_id}")
def get_style(style_id: str):
    pack = get_style_pack(style_id)
    return {
        "ok": True,
        "item": {
            "meta": pack.meta.model_dump(),
            "constraints": pack.constraints.model_dump(),
            # Do not expose full templates by default (security/UX).
            "initial_situation": pack.initial_situation,
        },
    }


@router.post("/select")
def select_style(body: SelectStyleBody = Body(...)):
    # Validate style existence
    if body.style_id not in STYLE_PACKS:
        raise HTTPException(status_code=404, detail="style_id not found")
    # Upsert user's current style
    d = USERS.setdefault(body.user_id, {})
    d["style_id"] = body.style_id
    return {"ok": True, "style_id": body.style_id}


@router.get("/{style_id}/raw")
def get_style_full(style_id: str):
    pack = get_style_pack(style_id)
    return {"ok": True, "item": pack.model_dump()}


@router.put("/{style_id}")
def update_style(style_id: str, payload: StylePack):
    pack = StylePack.model_validate(payload)
    pack.meta.id = style_id
    register_style(pack)
    return {"ok": True, "item": pack.model_dump()}


@lab_router.get("/styles-lab", response_class=HTMLResponse)
def style_lab_ui():
    try:
        html = LAB_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - setup issue
        raise HTTPException(status_code=500, detail="Style lab template missing") from exc
    return HTMLResponse(html)

