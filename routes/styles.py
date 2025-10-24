# routes/styles.py
# FastAPI router for style pack endpoints only.

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from style_packs import get_style_pack, list_style_meta, STYLE_PACKS
from stores import USERS

router = APIRouter(prefix="/api/styles", tags=["styles"])

class SelectStyleBody(BaseModel):
    user_id: str
    style_id: str

@router.get("")
def get_styles():
    # Return list of style meta with initial situation details
    items = []
    for meta in list_style_meta():
        pack = get_style_pack(meta.id)
        item = meta.dict()
        item["initial_situation"] = pack.initial_situation
        items.append(item)
    return {"ok": True, "items": items}

@router.get("/{style_id}")
def get_style(style_id: str):
    pack = get_style_pack(style_id)
    return {
        "ok": True,
        "item": {
            "meta": pack.meta.dict(),
            "constraints": pack.constraints.dict(),
            # Do not expose full templates by default (security/UX).
            "initial_situation": pack.initial_situation,
        }
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