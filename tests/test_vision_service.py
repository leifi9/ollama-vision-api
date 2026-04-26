"""
Tests fuer vision_service: _parse_json_response, validate_image, analyze_image,
check_ollama_health, list_vision_models.
Ollama-HTTP-Calls werden mit unittest.mock gemockt.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services.vision_service import (
    ALLOWED_EXTENSIONS,
    _parse_json_response,
    analyze_image,
    check_ollama_health,
    list_vision_models,
    validate_image,
)


# ─── _parse_json_response ───────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        payload = {"name": "mug", "keywords": ["cup"], "description": "A mug.", "search_query": "mug stl"}
        result = _parse_json_response(json.dumps(payload))
        assert result["name"] == "mug"
        assert result["keywords"] == ["cup"]

    def test_backtick_json_block(self):
        text = '```json\n{"name": "gear", "keywords": [], "description": "A gear.", "search_query": "gear"}\n```'
        result = _parse_json_response(text)
        assert result["name"] == "gear"

    def test_backtick_no_lang(self):
        text = '```\n{"name": "bolt", "keywords": [], "description": "x", "search_query": "bolt"}\n```'
        result = _parse_json_response(text)
        assert result["name"] == "bolt"

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"name": "bracket", "keywords": [], "description": "d", "search_query": "s"} Done.'
        result = _parse_json_response(text)
        assert result["name"] == "bracket"

    def test_fallback_raw_text(self):
        result = _parse_json_response("This is not JSON at all.")
        assert result["name"] == "unknown"
        assert result["keywords"] == []
        assert "This is not JSON" in result["description"]

    def test_empty_string_fallback(self):
        result = _parse_json_response("")
        assert result["name"] == "unknown"

    def test_invalid_json_fallback(self):
        result = _parse_json_response("{broken json")
        assert result["name"] == "unknown"


# ─── validate_image ────────────────────────────────────────────────

class TestValidateImage:
    def test_valid_jpg(self):
        assert validate_image("photo.jpg", 100) is None

    def test_valid_png(self):
        assert validate_image("scan.PNG", 100) is None

    def test_valid_webp(self):
        assert validate_image("img.webp", 500) is None

    def test_invalid_extension(self):
        err = validate_image("doc.pdf", 100)
        assert err is not None
        assert ".pdf" in err

    def test_file_too_large(self):
        # default max is 20 MB
        err = validate_image("big.jpg", 21 * 1024 * 1024)
        assert err is not None
        assert "groß" in err or "MB" in err

    def test_exactly_at_limit(self):
        # 20 MB exactly should be ok
        from backend.config import settings
        assert validate_image("edge.jpg", settings.max_file_size_bytes) is None

    def test_all_allowed_extensions(self):
        for ext in ALLOWED_EXTENSIONS:
            assert validate_image(f"file{ext}", 100) is None, f"Failed for {ext}"


# ─── analyze_image ─────────────────────────────────────────────────

GOOD_JSON = json.dumps({
    "name": "wrench",
    "keywords": ["tool", "repair"],
    "description": "A metal wrench.",
    "search_query": "wrench STL",
})


def _make_tmp_image() -> Path:
    """Erstellt eine echte (minimale) Temp-Bilddatei."""
    # 1x1 PNG bytes
    import base64
    TINY_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(TINY_PNG)
    tmp.close()
    return Path(tmp.name)


class TestAnalyzeImage:
    def test_success(self):
        tmp = _make_tmp_image()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "llava:7b",
            "message": {"content": GOOD_JSON},
            "total_duration": 1_000_000_000,
            "eval_count": 42,
        }
        with patch("backend.services.vision_service.requests.post", return_value=mock_resp):
            result = analyze_image(tmp)
        assert result["name"] == "wrench"
        assert result["_meta"]["model"] == "llava:7b"
        assert result["_meta"]["total_duration_ms"] == 1000
        tmp.unlink()

    def test_connection_error(self):
        import requests as req_lib
        tmp = _make_tmp_image()
        with patch("backend.services.vision_service.requests.post", side_effect=req_lib.ConnectionError):
            with pytest.raises(ConnectionError, match="Ollama nicht erreichbar"):
                analyze_image(tmp)
        tmp.unlink()

    def test_timeout(self):
        import requests as req_lib
        tmp = _make_tmp_image()
        with patch("backend.services.vision_service.requests.post", side_effect=req_lib.Timeout):
            with pytest.raises(RuntimeError, match="Timeout"):
                analyze_image(tmp)
        tmp.unlink()

    def test_http_error(self):
        tmp = _make_tmp_image()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("backend.services.vision_service.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                analyze_image(tmp)
        tmp.unlink()

    def test_empty_answer(self):
        tmp = _make_tmp_image()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"model": "llava:7b", "message": {"content": ""}}
        with patch("backend.services.vision_service.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="leer"):
                analyze_image(tmp)
        tmp.unlink()

    def test_custom_model(self):
        tmp = _make_tmp_image()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "llava:13b",
            "message": {"content": GOOD_JSON},
            "total_duration": 0,
            "eval_count": 0,
        }
        captured = {}
        def capture_post(url, json=None, timeout=None):
            captured["payload"] = json
            return mock_resp
        with patch("backend.services.vision_service.requests.post", side_effect=capture_post):
            analyze_image(tmp, model="llava:13b")
        assert captured["payload"]["model"] == "llava:13b"
        tmp.unlink()


# ─── check_ollama_health ───────────────────────────────────────────

class TestCheckOllamaHealth:
    def test_model_ready(self):
        from backend.config import settings
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": settings.vision_model}, {"name": "llama3"}]}
        with patch("backend.services.vision_service.requests.get", return_value=mock_resp):
            result = check_ollama_health()
        assert result["status"] == "ok"
        assert result["model_available"] is True

    def test_model_missing(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        with patch("backend.services.vision_service.requests.get", return_value=mock_resp):
            result = check_ollama_health()
        assert result["status"] == "model_missing"
        assert result["model_available"] is False

    def test_offline(self):
        import requests as req_lib
        with patch("backend.services.vision_service.requests.get", side_effect=req_lib.ConnectionError):
            result = check_ollama_health()
        assert result["status"] == "offline"

    def test_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("backend.services.vision_service.requests.get", return_value=mock_resp):
            result = check_ollama_health()
        assert result["status"] == "error"


# ─── list_vision_models ────────────────────────────────────────────

class TestListVisionModels:
    def test_returns_models(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llava:7b"}, {"name": "llama3"}]}
        with patch("backend.services.vision_service.requests.get", return_value=mock_resp):
            models = list_vision_models()
        assert "llava:7b" in models
        assert "llama3" in models

    def test_offline_returns_empty(self):
        import requests as req_lib
        with patch("backend.services.vision_service.requests.get", side_effect=req_lib.ConnectionError):
            models = list_vision_models()
        assert models == []

    def test_http_error_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("backend.services.vision_service.requests.get", return_value=mock_resp):
            models = list_vision_models()
        assert models == []
