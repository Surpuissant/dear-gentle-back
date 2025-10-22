"""Serve API reference content without mixing it with the business logic."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse


DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "api_reference.md"

try:
    DOC_CONTENT = DOC_PATH.read_text(encoding="utf-8")
except FileNotFoundError as exc:  # pragma: no cover - configuration error
    raise RuntimeError(
        "Missing documentation file at docs/api_reference.md"
    ) from exc


router = APIRouter(prefix="/api/docs", tags=["documentation"])


@router.get("", response_class=PlainTextResponse, summary="API reference (Markdown)")
async def get_api_reference_markdown() -> PlainTextResponse:
    """Return the API reference as Markdown for human readers."""
    return PlainTextResponse(DOC_CONTENT, media_type="text/markdown; charset=utf-8")


def _parse_sections(markdown: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    lines: List[str] = []

    for line in markdown.splitlines():
        if line.startswith("### "):
            if current is not None:
                current["content"] = "\n".join(lines).strip()
                sections.append(current)
            current = {"title": line.removeprefix("### ").strip()}
            lines = []
            continue
        if current is None:
            # Ignore everything before the first section heading
            continue
        lines.append(line)

    if current is not None:
        current["content"] = "\n".join(lines).strip()
        sections.append(current)

    return sections


@router.get(
    "/structured",
    summary="API reference as structured JSON",
    response_description="A list of high level sections with their Markdown body.",
)
async def get_api_reference_structured() -> Dict[str, List[Dict[str, str]]]:
    """Expose the same documentation as machine-friendly JSON."""
    sections = _parse_sections(DOC_CONTENT)
    if not sections:
        raise HTTPException(status_code=500, detail="Documentation is empty")
    return {"title": "Dear Gentle API", "sections": sections}

