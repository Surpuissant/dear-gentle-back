from fastapi import HTTPException, APIRouter

from app import BOOKS, perform_chapter_edit
from models import ChapterEditRequest
from stores import CHAPTERS, CHAPTER_VERSIONS

router = APIRouter(prefix="/api/chapters", tags=["chapters"])

@router.get("")
def list_chapters(book_id: str):
    items = [c for c in CHAPTERS.values() if c.book_id == book_id]
    items = sorted(items, key=lambda x: x.index)
    return {"ok": True, "items": items}


@router.get("/{chapter_id}")
def get_chapter(chapter_id: str):
    ch = CHAPTERS.get(chapter_id)
    if not ch:
        raise HTTPException(status_code=404, detail="chapter not found")
    return {"ok": True, "item": ch, "versions": CHAPTER_VERSIONS.get(chapter_id, [])}

@router.post("/{chapter_id}/edit")
def edit_chapter(chapter_id: str, req: ChapterEditRequest):
    ch = CHAPTERS.get(chapter_id)
    if not ch:
        raise HTTPException(status_code=404, detail="chapter not found")
    book = BOOKS.get(ch.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    updated, used_facets, used_mem_ids = perform_chapter_edit(
        chapter=ch,
        user_id=req.user_id,
        edit_instruction=req.edit_instruction,
    )

    return {"ok": True, "item": updated, "used_facets": used_facets, "used_memory_ids": used_mem_ids}
