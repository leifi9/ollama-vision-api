"""
Integrationstests fuer die FastAPI-Router:
  GET  /api/health
  GET  /api/models
  POST /api/analyze

Nutzt FastAPI TestClient (synchron, kein Ollama noetig — alles gemockt).
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── App-Instanz mit gemockter DB (temp-Datei) ─────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient mit isolierter Temp-DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        upload_path = Path(tmpdir) / "uploads"
        upload_path.mkdir()

        with (
            patch("backend.config.settings.database_path", str(db_path)),
            patch("backend.config.settings.persistent_upload_dir", str(upload_path)),
        ):
            # Late import so patches take effect before app initialization
            from backend.main import app
            from backend.database import init_db

            # Re-init DB with patched path
            with patch("backend.database.settings") as ms:
                ms.database_file = db_path
                init_db()

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


# Tiny valid PNG (1x1 pixel)
import base64
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

_GOOD_ANALYSIS = {
    "name": "cube",
    "keywords": ["shape", "geometric"],
    "description": "A simple cube.",
    "search_query": "cube STL",
    "_meta": {"model": "llava:7b", "total_duration_ms": 200},
}


# ── GET /api/health ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        mock_health = {
            "status": "ok",
            "ollama_url": "http://localhost:11434",
            "vision_model": "llava:7b",
            "model_available": True,
            "available_models": ["llava:7b"],
        }
        with patch("backend.routers.analyze.check_ollama_health", return_value=mock_health):
            r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_offline(self, client):
        mock_health = {"status": "offline", "detail": "not reachable"}
        with patch("backend.routers.analyze.check_ollama_health", return_value=mock_health):
            r = client.get("/api/health")
        # Status code 503 when offline
        assert r.status_code == 503

    def test_health_model_missing(self, client):
        mock_health = {
            "status": "model_missing",
            "ollama_url": "http://localhost:11434",
            "vision_model": "llava:7b",
            "model_available": False,
            "available_models": [],
        }
        with patch("backend.routers.analyze.check_ollama_health", return_value=mock_health):
            r = client.get("/api/health")
        assert r.status_code == 503


# ── GET /api/models ────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_returns_list(self, client):
        with patch("backend.routers.analyze.list_vision_models", return_value=["llava:7b", "llama3"]):
            r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "llava:7b" in data["models"]

    def test_empty_list_when_offline(self, client):
        with patch("backend.routers.analyze.list_vision_models", return_value=[]):
            r = client.get("/api/models")
        assert r.status_code == 200
        assert r.json()["models"] == []


# ── POST /api/analyze ──────────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    def _upload(self, client, content=_TINY_PNG, filename="test.png", extra_patches=None):
        patches = extra_patches or {}
        with patch("backend.routers.analyze.analyze_image", return_value=dict(_GOOD_ANALYSIS)) as mock_analyze:
            with patch("backend.routers.analyze.persist_analyzed_image", return_value=(1, 1, "stored.png")):
                r = client.post(
                    "/api/analyze",
                    files={"file": (filename, io.BytesIO(content), "image/png")},
                )
        return r, mock_analyze

    def test_success_returns_200(self, client):
        r, _ = self._upload(client)
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "cube"
        assert "keywords" in data

    def test_success_includes_ids_when_persisted(self, client):
        r, _ = self._upload(client)
        data = r.json()
        assert "image_id" in data
        assert "analysis_id" in data
        assert "stored_as" in data

    def test_invalid_extension_returns_400(self, client):
        r = client.post(
            "/api/analyze",
            files={"file": ("doc.pdf", io.BytesIO(b"fake"), "application/pdf")},
        )
        assert r.status_code == 400
        assert ".pdf" in r.json()["detail"]

    def test_ollama_offline_returns_503(self, client):
        import requests as req_lib
        with patch("backend.routers.analyze.analyze_image", side_effect=ConnectionError("Ollama offline")):
            r = client.post(
                "/api/analyze",
                files={"file": ("img.png", io.BytesIO(_TINY_PNG), "image/png")},
            )
        assert r.status_code == 503

    def test_ollama_runtime_error_returns_502(self, client):
        with patch("backend.routers.analyze.analyze_image", side_effect=RuntimeError("model error")):
            r = client.post(
                "/api/analyze",
                files={"file": ("img.png", io.BytesIO(_TINY_PNG), "image/png")},
            )
        assert r.status_code == 502

    def test_no_persist_skips_db(self, client):
        with patch("backend.routers.analyze.analyze_image", return_value=dict(_GOOD_ANALYSIS)) as mock_analyze:
            with patch("backend.routers.analyze.persist_analyzed_image") as mock_persist:
                r = client.post(
                    "/api/analyze?persist=false",
                    files={"file": ("img.png", io.BytesIO(_TINY_PNG), "image/png")},
                )
                mock_persist.assert_not_called()
        assert r.status_code == 200
        assert "image_id" not in r.json()

    def test_custom_prompt_forwarded(self, client):
        captured = {}
        def capture(path, prompt=None, model=None):
            captured["prompt"] = prompt
            return dict(_GOOD_ANALYSIS)
        with patch("backend.routers.analyze.analyze_image", side_effect=capture):
            with patch("backend.routers.analyze.persist_analyzed_image", return_value=(1, 1, "x.png")):
                client.post(
                    "/api/analyze",
                    data={"prompt": "my custom prompt"},
                    files={"file": ("img.png", io.BytesIO(_TINY_PNG), "image/png")},
                )
        assert captured.get("prompt") == "my custom prompt"
