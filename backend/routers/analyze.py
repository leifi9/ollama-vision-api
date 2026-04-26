"""
API-Router für Bildanalyse-Endpunkte.
"""

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.config import settings
from backend.database import get_conn
from backend.services.gallery_service import persist_analyzed_image
from backend.services.vision_service import (
    analyze_image,
    check_ollama_health,
    list_vision_models,
    validate_image,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Vision"])


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Bilddatei (JPG, PNG, WebP, GIF)"),
    prompt: str | None = Form(
        default=None,
        description="Optionaler Custom-Prompt. Leer = Standard-3D-Print-Analyse.",
    ),
    persist: bool = Query(
        default=True,
        description="Bild + Analyse dauerhaft in die Galerie speichern.",
    ),
):
    """
    Lädt ein Bild hoch und analysiert es via Ollama + LLaVA.

    Gibt strukturierte JSON-Daten zurück (name, keywords, description,
    search_query). Bei persist=true wird das Bild zusätzlich in der
    Galerie gespeichert und im Response werden image_id + analysis_id
    mitgeliefert.
    """
    content = await file.read()
    original_name = file.filename or "unknown"

    error = validate_image(original_name, len(content))
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Temp file for vision_service (reads from disk).
    ext = Path(original_name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = analyze_image(tmp_path, prompt)
        result["filename"] = original_name

        if persist:
            with get_conn() as conn:
                image_id, analysis_id, stored = persist_analyzed_image(
                    conn,
                    image_bytes=content,
                    original_name=original_name,
                    analysis=result,
                    prompt_used=prompt,
                )
            result["image_id"] = image_id
            result["analysis_id"] = analysis_id
            result["stored_as"] = stored

        return result

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Unerwarteter Fehler bei Bildanalyse")
        raise HTTPException(status_code=500, detail=f"Interner Fehler: {e}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@router.get("/health")
async def health():
    """Health-Check: Prüft Ollama-Verbindung und Modell-Verfügbarkeit."""
    from fastapi.responses import JSONResponse
    status = check_ollama_health()
    code = 200 if status["status"] == "ok" else 503
    return JSONResponse(content=status, status_code=code)


@router.get("/models")
async def models():
    """Listet alle lokal verfügbaren Ollama-Modelle auf."""
    return {"models": list_vision_models()}
