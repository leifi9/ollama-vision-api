"""
Gallery API endpoints: list, search, detail, delete, serve image files.

Batch upload and re-analyze live in this module too but are added in later phases.
All responses follow the unified error envelope: {"error": {"code","message","hint"}}.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database import (
    delete_image,
    get_analyses_for_image,
    get_conn,
    get_image,
    get_latest_analysis,
    insert_analysis,
    list_images,
    search_analyses,
)
from backend.services.gallery_service import persist_analyzed_image
from backend.services.vision_service import analyze_image, validate_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Gallery"])


# ── Error helpers ─────────────────────────────────────────────────────────

def error_response(
    status_code: int,
    code: str,
    message: str,
    hint: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "hint": hint}},
    )


# ── Response models (for Swagger) ─────────────────────────────────────────

class AnalysisSummary(BaseModel):
    id: int
    name: str | None
    keywords: list[str]
    description: str | None
    search_query: str | None
    model_used: str | None
    created_at: str


class AnalysisFull(AnalysisSummary):
    image_id: int
    prompt_used: str | None
    duration_ms: int | None


class GalleryItem(BaseModel):
    id: int
    filename: str
    original_name: str
    file_size: int | None
    created_at: str
    latest_analysis: AnalysisSummary | None = None


class GalleryListResponse(BaseModel):
    items: list[GalleryItem]
    total: int
    page: int
    per_page: int


class ImageDetailResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    file_size: int | None
    created_at: str
    analyses: list[AnalysisFull]


class SearchResponse(BaseModel):
    query: str
    items: list[GalleryItem]
    count: int


class DeleteResponse(BaseModel):
    deleted: int
    filename: str


class ErrorBody(BaseModel):
    code: str
    message: str
    hint: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get(
    "/gallery",
    response_model=GalleryListResponse,
    summary="List gallery images with their latest analysis",
)
async def list_gallery(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: str = Query("created_at", pattern="^(created_at|id|original_name)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    with get_conn() as conn:
        items, total = list_images(
            conn, page=page, per_page=per_page, sort=sort, order=order
        )
    return {"items": items, "total": total, "page": page, "per_page": per_page}


@router.get(
    "/gallery/search",
    response_model=SearchResponse,
    summary="Full-text search across all analyses",
)
async def gallery_search(
    q: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(50, ge=1, le=200),
):
    with get_conn() as conn:
        items = search_analyses(conn, q, limit=limit)
    return {"query": q, "items": items, "count": len(items)}


@router.get(
    "/gallery/{image_id}",
    response_model=ImageDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get one image with all its analyses",
)
async def gallery_detail(image_id: int):
    with get_conn() as conn:
        img = get_image(conn, image_id)
        if img is None:
            return error_response(
                404,
                "image_not_found",
                f"No image with id {image_id}",
                hint="List available images via GET /api/gallery",
            )
        analyses = get_analyses_for_image(conn, image_id)
    return {
        "id": img["id"],
        "filename": img["filename"],
        "original_name": img["original_name"],
        "file_size": img["file_size"],
        "created_at": img["created_at"],
        "analyses": analyses,
    }


@router.delete(
    "/gallery/{image_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete an image and all its analyses",
)
async def gallery_delete(image_id: int):
    with get_conn() as conn:
        row = delete_image(conn, image_id)
        if row is None:
            return error_response(
                404,
                "image_not_found",
                f"No image with id {image_id}",
            )
        filename = row["filename"]
        file_path = row["file_path"]

    # File cleanup outside the DB transaction — best-effort, log on failure.
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except OSError as e:
        logger.warning(f"Konnte Bilddatei nicht löschen ({file_path}): {e}")

    return {"deleted": image_id, "filename": filename}


@router.post(
    "/gallery/batch",
    summary="Upload multiple images, analyze sequentially, stream progress via SSE",
    response_class=StreamingResponse,
)
async def gallery_batch(
    files: list[UploadFile] = File(..., description="1..N image files"),
    prompt: str | None = Query(
        default=None,
        description="Optional custom prompt applied to every image in the batch.",
    ),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch too large: {len(files)} files "
                f"(max {settings.max_batch_size})"
            ),
        )

    # Read all file bytes up-front so we can stream events without awaiting
    # the client mid-stream. Small cost, much simpler flow.
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        payloads.append((f.filename or "unknown", await f.read()))

    total = len(payloads)

    async def stream() -> AsyncIterator[bytes]:
        yield _sse({"type": "batch_start", "total": total})

        succeeded = 0
        failed = 0

        for index, (original_name, content) in enumerate(payloads):
            # Per-image validation (format + size).
            err = validate_image(original_name, len(content))
            if err:
                failed += 1
                yield _sse(
                    {
                        "type": "image_error",
                        "index": index,
                        "filename": original_name,
                        "error": {
                            "code": "invalid_image",
                            "message": err,
                            "hint": (
                                "Allowed: jpg, jpeg, png, gif, webp, bmp; "
                                f"max {settings.max_file_size_mb} MB"
                            ),
                        },
                    }
                )
                continue

            # Write to a temp file for vision_service (which reads from disk).
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(original_name).suffix.lower() or ".jpg",
                ) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)

                try:
                    result = analyze_image(tmp_path, prompt)
                except ConnectionError as e:
                    yield _sse(
                        {
                            "type": "batch_aborted",
                            "reason": {
                                "code": "ollama_unreachable",
                                "message": str(e),
                                "hint": "Start Ollama with `ollama serve`.",
                            },
                            "completed_count": succeeded,
                        }
                    )
                    return
                except RuntimeError as e:
                    yield _sse(
                        {
                            "type": "batch_aborted",
                            "reason": {
                                "code": "ollama_error",
                                "message": str(e),
                                "hint": (
                                    "Upstream Ollama failed. "
                                    "Check `ollama serve` logs."
                                ),
                            },
                            "completed_count": succeeded,
                        }
                    )
                    return

                with get_conn() as conn:
                    image_id, analysis_id, stored = persist_analyzed_image(
                        conn,
                        image_bytes=content,
                        original_name=original_name,
                        analysis=result,
                        prompt_used=prompt,
                    )

                succeeded += 1
                yield _sse(
                    {
                        "type": "image_done",
                        "index": index,
                        "filename": original_name,
                        "image_id": image_id,
                        "analysis_id": analysis_id,
                        "stored_as": stored,
                        "result": {
                            k: v for k, v in result.items() if k != "_meta"
                        },
                        "meta": result.get("_meta", {}),
                    }
                )
            except Exception as e:
                logger.exception("Unexpected batch-item error")
                failed += 1
                yield _sse(
                    {
                        "type": "image_error",
                        "index": index,
                        "filename": original_name,
                        "error": {
                            "code": "internal_error",
                            "message": str(e),
                            "hint": None,
                        },
                    }
                )
            finally:
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        yield _sse(
            {
                "type": "batch_complete",
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
            }
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class ReanalyzeRequest(BaseModel):
    prompt: str | None = Field(
        default=None,
        description="Custom prompt. Null → reuse prompt from last analysis.",
    )
    model: str | None = Field(
        default=None,
        description="Ollama model name. Null → reuse model from last analysis.",
    )


@router.post(
    "/gallery/{image_id}/reanalyze",
    response_model=AnalysisFull,
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Re-analyze a stored image with an optional new prompt or model",
)
async def gallery_reanalyze(image_id: int, body: ReanalyzeRequest = ReanalyzeRequest()):
    with get_conn() as conn:
        img = get_image(conn, image_id)
        if img is None:
            return error_response(
                404,
                "image_not_found",
                f"No image with id {image_id}",
                hint="List available images via GET /api/gallery",
            )
        last = get_latest_analysis(conn, image_id)

    # Resolve prompt / model: explicit request → fallback to last analysis → config default
    prompt = body.prompt or (last["prompt_used"] if last else None)
    model = body.model or (last["model_used"] if last else None)

    file_path = Path(img["file_path"])
    if not file_path.is_file():
        return error_response(
            404,
            "image_file_missing",
            f"Image file for id {image_id} no longer exists on disk",
            hint="The image may have been deleted from the filesystem manually.",
        )

    try:
        result = analyze_image(file_path, prompt=prompt, model=model)
    except ConnectionError as e:
        return error_response(503, "ollama_unreachable", str(e), hint="Run `ollama serve`.")
    except RuntimeError as e:
        return error_response(503, "ollama_error", str(e))

    meta = result.get("_meta", {}) or {}

    with get_conn() as conn:
        analysis_id = insert_analysis(
            conn,
            image_id=image_id,
            name=result.get("name"),
            keywords=result.get("keywords") or [],
            description=result.get("description"),
            search_query=result.get("search_query"),
            prompt_used=prompt,
            model_used=meta.get("model"),
            duration_ms=meta.get("total_duration_ms"),
        )
        new_analysis = get_latest_analysis(conn, image_id)

    return new_analysis


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.get(
    "/images/{filename}",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    summary="Serve a stored image file",
)
async def serve_image(filename: str):
    # Path traversal protection: resolve + verify parent is the image store.
    store = settings.persistent_upload_path
    candidate = (store / filename).resolve()
    try:
        candidate.relative_to(store)
    except ValueError:
        return error_response(
            400,
            "invalid_filename",
            "Filename resolves outside the image store",
            hint="Filenames must not contain path separators",
        )
    if not candidate.is_file():
        return error_response(
            404, "image_file_missing", f"File '{filename}' not found on disk"
        )
    return FileResponse(candidate)
