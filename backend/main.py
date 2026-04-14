"""
Ollama Vision API – Hauptapplikation.
Lokale Bilderkennung via LLaVA / Ollama.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.database import init_db
from backend.routers.analyze import router as analyze_router
from backend.routers.gallery import router as gallery_router
from backend.services.vision_service import check_ollama_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown Events."""
    # ── Startup ──
    logger.info("═" * 60)
    logger.info("  Ollama Vision API wird gestartet")
    logger.info(f"  Ollama URL:    {settings.ollama_base_url}")
    logger.info(f"  Vision Model:  {settings.vision_model}")
    logger.info(f"  Max Upload:    {settings.max_file_size_mb} MB")
    logger.info(f"  DB File:       {settings.database_file}")
    logger.info(f"  Image Store:   {settings.persistent_upload_path}")
    logger.info("═" * 60)

    init_db()

    health = check_ollama_health()
    if health["status"] == "offline":
        logger.warning(
            "⚠  Ollama ist NICHT erreichbar! "
            "Starte Ollama mit 'ollama serve'."
        )
    elif health["status"] == "model_missing":
        logger.warning(
            f"⚠  Modell '{settings.vision_model}' nicht gefunden. "
            f"Lade es mit: ollama pull {settings.vision_model}"
        )
    else:
        logger.info(f"✓  Ollama verbunden, Modell '{settings.vision_model}' bereit.")

    yield

    # ── Shutdown ──
    logger.info("Ollama Vision API wird beendet.")


app = FastAPI(
    title="Ollama Vision API",
    description=(
        "Self-hosted Bildanalyse via Ollama + LLaVA. "
        "Erkennt Objekte und gibt strukturierte JSON-Daten zurück."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Router einbinden
app.include_router(analyze_router)
app.include_router(gallery_router)

# Statische Dateien (Web-UI)
app.mount(
    "/static",
    StaticFiles(directory="backend/static"),
    name="static",
)


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Web-UI ausliefern."""
    return FileResponse("backend/static/index.html")
