"""
Tests fuer gallery_service.persist_analyzed_image und database CRUD:
insert_image, get_image, delete_image, list_images, insert_analysis,
get_latest_analysis, get_analyses_for_image, search_analyses.

Alle Tests verwenden eine In-Memory / Tmp-SQLite-DB — kein echter Disk-State.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.database import (
    delete_image,
    get_analyses_for_image,
    get_image,
    get_latest_analysis,
    init_db,
    insert_analysis,
    insert_image,
    list_images,
    search_analyses,
)
from backend.services.gallery_service import persist_analyzed_image


# ─── Fixture: frische In-Memory DB ────────────────────────────────────────────

@pytest.fixture
def conn():
    """Gibt eine frische SQLite-Verbindung mit vollstaendigem Schema zurueck."""
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    # Schema aus init_db replizieren — wir rufen init_db mit gemocktem settings auf
    with patch("backend.database.settings") as mock_settings:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        mock_settings.database_file = Path(tmp.name)
        # init_db oeffnet eigene Verbindung — wir bauen Schema direkt nach
    c.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            name TEXT,
            keywords TEXT,
            description TEXT,
            search_query TEXT,
            prompt_used TEXT,
            model_used TEXT,
            duration_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS analyses_fts USING fts5(
            name, keywords, description, search_query,
            content='analyses', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS analyses_ai AFTER INSERT ON analyses BEGIN
            INSERT INTO analyses_fts(rowid, name, keywords, description, search_query)
            VALUES (
                new.id, new.name,
                REPLACE(REPLACE(COALESCE(new.keywords, ''), '[', ''), ']', ''),
                new.description, new.search_query
            );
        END;
        CREATE TRIGGER IF NOT EXISTS analyses_ad AFTER DELETE ON analyses BEGIN
            INSERT INTO analyses_fts(analyses_fts, rowid, name, keywords, description, search_query)
            VALUES ('delete', old.id, old.name,
                REPLACE(REPLACE(COALESCE(old.keywords, ''), '[', ''), ']', ''),
                old.description, old.search_query);
        END;
    """)
    c.commit()
    yield c
    c.close()


# ─── insert_image / get_image ──────────────────────────────────────────────────

class TestInsertGetImage:
    def test_insert_returns_id(self, conn):
        img_id = insert_image(conn, "abc.png", "photo.png", "/tmp/abc.png", 1024)
        assert isinstance(img_id, int)
        assert img_id >= 1

    def test_get_returns_row(self, conn):
        img_id = insert_image(conn, "x.jpg", "original.jpg", "/tmp/x.jpg", 512)
        row = get_image(conn, img_id)
        assert row is not None
        assert row["filename"] == "x.jpg"
        assert row["original_name"] == "original.jpg"
        assert row["file_size"] == 512

    def test_get_missing_returns_none(self, conn):
        assert get_image(conn, 9999) is None

    def test_multiple_images(self, conn):
        id1 = insert_image(conn, "a.png", "a.png", "/tmp/a.png", 100)
        id2 = insert_image(conn, "b.png", "b.png", "/tmp/b.png", 200)
        assert id1 != id2
        assert get_image(conn, id1)["filename"] == "a.png"
        assert get_image(conn, id2)["filename"] == "b.png"


# ─── delete_image ──────────────────────────────────────────────────────────────

class TestDeleteImage:
    def test_delete_returns_row(self, conn):
        img_id = insert_image(conn, "del.jpg", "del.jpg", "/tmp/del.jpg", 300)
        row = delete_image(conn, img_id)
        assert row is not None
        assert row["id"] == img_id

    def test_delete_removes_from_db(self, conn):
        img_id = insert_image(conn, "gone.jpg", "gone.jpg", "/tmp/gone.jpg", 50)
        delete_image(conn, img_id)
        assert get_image(conn, img_id) is None

    def test_delete_nonexistent_returns_none(self, conn):
        assert delete_image(conn, 9999) is None

    def test_delete_cascades_analyses(self, conn):
        img_id = insert_image(conn, "c.png", "c.png", "/tmp/c.png", 100)
        insert_analysis(conn, img_id, "tool", ["x"], None, None, None, None, None)
        delete_image(conn, img_id)
        rows = conn.execute("SELECT * FROM analyses WHERE image_id = ?", (img_id,)).fetchall()
        assert rows == []


# ─── list_images ───────────────────────────────────────────────────────────────

class TestListImages:
    def test_empty_db(self, conn):
        items, total = list_images(conn)
        assert items == []
        assert total == 0

    def test_single_image(self, conn):
        insert_image(conn, "s.png", "s.png", "/tmp/s.png", 10)
        items, total = list_images(conn)
        assert total == 1
        assert items[0]["filename"] == "s.png"

    def test_pagination(self, conn):
        for i in range(5):
            insert_image(conn, f"{i}.png", f"{i}.png", f"/tmp/{i}.png", i * 10)
        items, total = list_images(conn, page=1, per_page=3)
        assert total == 5
        assert len(items) == 3

    def test_page_2(self, conn):
        for i in range(5):
            insert_image(conn, f"{i}.png", f"{i}.png", f"/tmp/{i}.png", i * 10)
        items, total = list_images(conn, page=2, per_page=3)
        assert total == 5
        assert len(items) == 2

    def test_with_analysis(self, conn):
        img_id = insert_image(conn, "a.png", "a.png", "/tmp/a.png", 100)
        insert_analysis(conn, img_id, "wrench", ["tool"], None, None, None, None, None)
        items, _ = list_images(conn)
        assert items[0]["latest_analysis"] is not None
        assert items[0]["latest_analysis"]["name"] == "wrench"

    def test_invalid_sort_falls_back(self, conn):
        insert_image(conn, "z.png", "z.png", "/tmp/z.png", 1)
        # Should not raise even with invalid sort
        items, total = list_images(conn, sort="INJECTED", order="DROP")
        assert total == 1


# ─── insert_analysis / get_latest_analysis ────────────────────────────────────

class TestAnalysis:
    def _insert(self, conn, image_id, name, keywords=None, description=None):
        return insert_analysis(conn, image_id, name, keywords or [], description, None, None, None, None)

    def test_insert_and_get_latest(self, conn):
        img_id = insert_image(conn, "t.jpg", "t.jpg", "/tmp/t.jpg", 100)
        an_id = insert_analysis(
            conn, img_id, "gear", ["mechanical", "round"], "A gear.", "gear STL", "analyze this", "llava:7b", 450,
        )
        assert isinstance(an_id, int)
        latest = get_latest_analysis(conn, img_id)
        assert latest is not None
        assert latest["name"] == "gear"
        assert "mechanical" in latest["keywords"]

    def test_get_analyses_for_image(self, conn):
        img_id = insert_image(conn, "m.png", "m.png", "/tmp/m.png", 50)
        self._insert(conn, img_id, "bolt")
        self._insert(conn, img_id, "nut")
        rows = get_analyses_for_image(conn, img_id)
        assert len(rows) == 2
        names = {r["name"] for r in rows}
        assert names == {"bolt", "nut"}

    def test_latest_returns_newest(self, conn):
        img_id = insert_image(conn, "n.png", "n.png", "/tmp/n.png", 50)
        self._insert(conn, img_id, "first")
        self._insert(conn, img_id, "second")
        latest = get_latest_analysis(conn, img_id)
        assert latest["name"] == "second"

    def test_keywords_deserialized_as_list(self, conn):
        img_id = insert_image(conn, "k.png", "k.png", "/tmp/k.png", 50)
        self._insert(conn, img_id, "x", ["a", "b", "c"])
        latest = get_latest_analysis(conn, img_id)
        assert isinstance(latest["keywords"], list)
        assert latest["keywords"] == ["a", "b", "c"]

    def test_empty_keywords(self, conn):
        img_id = insert_image(conn, "e.png", "e.png", "/tmp/e.png", 50)
        self._insert(conn, img_id, "y", [])
        latest = get_latest_analysis(conn, img_id)
        assert latest["keywords"] == []

    def test_no_analysis_returns_none(self, conn):
        img_id = insert_image(conn, "no.png", "no.png", "/tmp/no.png", 50)
        assert get_latest_analysis(conn, img_id) is None


# ─── search_analyses ──────────────────────────────────────────────────────────

class TestSearchAnalyses:
    def _seed(self, conn):
        img1 = insert_image(conn, "1.png", "1.png", "/tmp/1.png", 100)
        img2 = insert_image(conn, "2.png", "2.png", "/tmp/2.png", 100)
        insert_analysis(conn, img1, "wrench", ["tool", "metal"], "A wrench for bolts.", None, None, None, None)
        insert_analysis(conn, img2, "gear", ["mechanical"], "A round gear.", None, None, None, None)
        conn.commit()

    def test_finds_by_name(self, conn):
        self._seed(conn)
        results = search_analyses(conn, "wrench")
        assert len(results) >= 1
        assert any(r["latest_analysis"]["name"] == "wrench" for r in results)

    def test_no_match_returns_empty(self, conn):
        self._seed(conn)
        results = search_analyses(conn, "zzznomatch999")
        assert results == []

    def test_finds_by_description(self, conn):
        self._seed(conn)
        results = search_analyses(conn, "bolts")
        assert any(r["latest_analysis"]["name"] == "wrench" for r in results)


# ─── persist_analyzed_image ───────────────────────────────────────────────────

class TestPersistAnalyzedImage:
    def test_writes_file_and_returns_ids(self, conn):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.services.gallery_service.settings") as mock_s:
                mock_s.persistent_upload_path = Path(tmpdir)

                analysis = {
                    "name": "cup",
                    "keywords": ["drink", "ceramic"],
                    "description": "A ceramic cup.",
                    "search_query": "cup STL",
                    "_meta": {"model": "llava:7b", "total_duration_ms": 300},
                }
                image_id, analysis_id, stored = persist_analyzed_image(
                    conn,
                    image_bytes=b"FAKEPNG",
                    original_name="photo.jpg",
                    analysis=analysis,
                    prompt_used="analyze",
                )

        assert isinstance(image_id, int)
        assert isinstance(analysis_id, int)
        assert stored.endswith(".jpg")

    def test_file_stored_with_uuid_name(self, conn):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.services.gallery_service.settings") as mock_s:
                mock_s.persistent_upload_path = Path(tmpdir)
                analysis = {"name": "x", "keywords": [], "_meta": {}}
                _, _, stored = persist_analyzed_image(
                    conn, b"DATA", "img.png", analysis, None
                )
                dest = Path(tmpdir) / stored
                assert dest.exists()
                assert dest.read_bytes() == b"DATA"

    def test_analysis_linked_to_image(self, conn):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.services.gallery_service.settings") as mock_s:
                mock_s.persistent_upload_path = Path(tmpdir)
                analysis = {
                    "name": "bracket",
                    "keywords": ["metal"],
                    "description": "desc",
                    "search_query": "bracket",
                    "_meta": {"model": "llava:7b", "total_duration_ms": 200},
                }
                image_id, analysis_id, _ = persist_analyzed_image(
                    conn, b"X", "b.png", analysis, "custom prompt"
                )
                latest = get_latest_analysis(conn, image_id)
                assert latest["name"] == "bracket"
                assert latest["model_used"] == "llava:7b"
