from typing import List

from fastapi import APIRouter, HTTPException, Body
from stores import INSTRUCTIONS

router = APIRouter(prefix="/api/instructions", tags=["instructions"])

# ----------------------------
# Instructions management API
# ----------------------------

@router.get("")
def list_instructions(user_id: str):
    """
    Return user's instruction overrides as an ordered list with indices.
    Response shape: { ok, items:[{index, rule_key, rule_value, active}] }
    """
    lst = INSTRUCTIONS.get(user_id, []) or []
    items = []
    for i, ins in enumerate(lst):
        # pydantic model -> dict; fallback if attributes missing
        d = ins.dict() if hasattr(ins, "dict") else {
            "rule_key": getattr(ins, "rule_key", "raw"),
            "rule_value": getattr(ins, "rule_value", ""),
            "active": getattr(ins, "active", True),
        }
        items.append({
            "index": i,
            "rule_key": d.get("rule_key", "raw"),
            "rule_value": d.get("rule_value", ""),
            "active": bool(d.get("active", True)),
        })
    return {"ok": True, "items": items}


@router.post("/toggle")
def toggle_instruction(user_id: str = Body(...), index: int = Body(...), active: bool = Body(...)):
    """
    Set active True/False by list index.
    """
    lst = INSTRUCTIONS.get(user_id, [])
    if index < 0 or index >= len(lst):
        raise HTTPException(status_code=404, detail="instruction index out of range")
    ins = lst[index]
    # tolerate missing field
    try:
        ins.active = active
    except Exception:
        # if it's a plain dict or incompatible, rebuild a dict-ish object
        setattr(ins, "active", active)
    return {"ok": True, "index": index, "active": active}


@router.delete("")
def delete_instructions(user_id: str, indexes: List[int] = Body(..., embed=True)):
    """
    Delete a list of instruction indices (sorted descending to keep indexes stable).
    """
    lst = INSTRUCTIONS.get(user_id, [])
    if not lst:
        return {"ok": True, "deleted": 0}
    to_del = sorted([i for i in indexes if 0 <= i < len(lst)], reverse=True)
    for i in to_del:
        lst.pop(i)
    INSTRUCTIONS[user_id] = lst
    return {"ok": True, "deleted": len(to_del)}
