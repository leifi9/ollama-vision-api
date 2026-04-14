"""
Gallery persistence: save an analyzed image + its analysis row to disk + DB.

Kept out of database.py so that DB code doesn't know about file I/O.
"""

import logging
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.database import insert_analysis, insert_image

logger = logging.getLogger(__name__)


def persist_analyzed_image(
    conn,
    image_bytes: bytes,
    original_name: str,
    analysis: dict[str, Any],
    prompt_used: str | None,
) -> tuple[int, int, str]:
    """
    Write the image bytes to the persistent store and insert image + analysis rows.

    Returns (image_id, analysis_id, stored_filename).
    """
    ext = Path(original_name).suffix.lower() or ".jpg"
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    dest = settings.persistent_upload_path / stored_filename
    with open(dest, "wb") as f:
        f.write(image_bytes)

    image_id = insert_image(
        conn,
        filename=stored_filename,
        original_name=original_name,
        file_path=str(dest),
        file_size=len(image_bytes),
    )

    meta = analysis.get("_meta", {}) or {}
    analysis_id = insert_analysis(
        conn,
        image_id=image_id,
        name=analysis.get("name"),
        keywords=analysis.get("keywords") or [],
        description=analysis.get("description"),
        search_query=analysis.get("search_query"),
        prompt_used=prompt_used,
        model_used=meta.get("model"),
        duration_ms=meta.get("total_duration_ms"),
    )
    return image_id, analysis_id, stored_filename
