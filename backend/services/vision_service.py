"""
Vision Service – Bildanalyse via Ollama + LLaVA.

Verwendet die offizielle Ollama REST API:
  POST /api/chat  mit Base64-kodierten Bildern im 'images'-Array.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Standard-Prompts ───────────────────────────────────────────────

DEFAULT_PROMPT = (
    "Analyze this image and return ONLY valid JSON (no markdown, no code "
    "fences, no explanation) with exactly these fields:\n"
    '  "name": short object name (string),\n'
    '  "keywords": 3-5 descriptive search keywords (array of strings),\n'
    '  "description": one-sentence description of the object (string),\n'
    '  "search_query": best search term to find this as a 3D printable '
    "STL file on Printables or Thingiverse (string)\n"
    "Respond with ONLY the JSON object."
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


# ─── Helper ─────────────────────────────────────────────────────────

def _encode_image(image_path: Path) -> str:
    """Liest eine Bilddatei und gibt Base64-String zurück."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_response(text: str) -> dict[str, Any]:
    """
    Versucht JSON aus der LLaVA-Antwort zu extrahieren.
    Behandelt typische Fälle: roher JSON, mit Backticks, mit Prä-/Suffix-Text.
    """
    cleaned = text.strip()

    # Fall 1: ```json ... ``` Blöcke entfernen
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    # Fall 2: Direkt parsen
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall 3: Erstes { ... } Objekt extrahieren
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: Rohtext als Beschreibung
    logger.warning("Konnte kein JSON aus LLaVA-Antwort parsen, verwende Fallback.")
    return {
        "name": "unknown",
        "keywords": [],
        "description": cleaned[:500],
        "search_query": "",
    }


# ─── Öffentliche API ────────────────────────────────────────────────

def validate_image(filename: str, file_size: int) -> str | None:
    """
    Validiert Dateiendung und -größe.
    Gibt None zurück wenn ok, sonst einen Fehlertext.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Nicht unterstütztes Format: {ext}. "
            f"Erlaubt: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if file_size > settings.max_file_size_bytes:
        return (
            f"Datei zu groß: {file_size / 1024 / 1024:.1f} MB. "
            f"Maximum: {settings.max_file_size_mb} MB"
        )
    return None


def analyze_image(
    image_path: Path,
    prompt: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Analysiert ein Bild via Ollama Chat API + LLaVA.

    Args:
        image_path: Pfad zur Bilddatei.
        prompt: Optionaler Custom-Prompt. Fallback: DEFAULT_PROMPT.
        model: Optionaler Modellname. Fallback: settings.vision_model.

    Returns:
        Dict mit name, keywords, description, search_query.

    Raises:
        ConnectionError: Ollama nicht erreichbar.
        RuntimeError: Ollama-Anfrage fehlgeschlagen.
    """
    image_b64 = _encode_image(image_path)
    used_prompt = prompt or DEFAULT_PROMPT
    used_model = model or settings.vision_model

    payload = {
        "model": used_model,
        "messages": [
            {
                "role": "user",
                "content": used_prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=180,
        )
    except requests.ConnectionError:
        raise ConnectionError(
            f"Ollama nicht erreichbar unter {settings.ollama_base_url}. "
            "Ist 'ollama serve' gestartet?"
        )
    except requests.Timeout:
        raise RuntimeError(
            "Ollama-Anfrage hat das Timeout überschritten (180s). "
            "Prüfe GPU-Auslastung oder verwende ein kleineres Modell."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama Fehler (HTTP {response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    answer_text = data.get("message", {}).get("content", "")

    if not answer_text:
        raise RuntimeError("Ollama hat eine leere Antwort zurückgegeben.")

    result = _parse_json_response(answer_text)

    # Metadaten anhängen
    result["_meta"] = {
        "model": data.get("model", settings.vision_model),
        "total_duration_ms": data.get("total_duration", 0) // 1_000_000,
        "eval_count": data.get("eval_count", 0),
    }

    return result


def check_ollama_health() -> dict[str, Any]:
    """Prüft ob Ollama erreichbar ist und das Modell geladen."""
    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if r.status_code != 200:
            return {"status": "error", "detail": f"HTTP {r.status_code}"}

        models = [m["name"] for m in r.json().get("models", [])]
        model_ready = any(settings.vision_model in m for m in models)

        return {
            "status": "ok" if model_ready else "model_missing",
            "ollama_url": settings.ollama_base_url,
            "vision_model": settings.vision_model,
            "model_available": model_ready,
            "available_models": models,
        }
    except requests.ConnectionError:
        return {
            "status": "offline",
            "detail": f"Ollama nicht erreichbar unter {settings.ollama_base_url}",
        }


def list_vision_models() -> list[str]:
    """Gibt alle lokal verfügbaren Ollama-Modelle zurück."""
    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except requests.ConnectionError:
        pass
    return []
