#!/usr/bin/env python3
"""
github_push.py — Push ollama-vision-api to GitHub via the Git Database API.

Flow (keine lokale git-Installation nötig):
  1. Auth       — GITHUB_TOKEN aus Umgebungsvariable
  2. Owner      — GET /user
  3. Repo       — sicherstellen dass er existiert, sonst anlegen
  4. Files      — Projektverzeichnis einsammeln (ignoriert: __pycache__, .env, *.pyc, uploads/)
  5. Blobs      — POST /git/blobs für jede Datei
  6. Tree       — POST /git/trees mit allen Blobs
  7. Commit     — POST /git/commits
  8. Ref        — POST (initial) oder PATCH (update) /git/refs/heads/main
"""

import base64
import json
import os
import sys
from pathlib import Path
from urllib import error as urllib_error
from urllib import request

# ── Konfiguration ─────────────────────────────────────────────────────────────

REPO_NAME      = "ollama-vision-api"
BRANCH         = "main"
COMMIT_MESSAGE = "feat: Batch Upload + Gallery Mode mit SQLite-Katalog"
PROJECT_DIR    = Path(__file__).parent

# Relativ-Pfade oder Teile davon, die nicht committet werden
IGNORE_NAMES = {
    "__pycache__", ".git", ".env", "uploads",
    "github_push.py",   # dieses Skript selbst nicht mitcommiten
}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".db", ".db-wal", ".db-shm"}

API_BASE = "https://api.github.com"

# Dateiendungen, die als Text (UTF-8) behandelt werden
TEXT_SUFFIXES = {
    ".py", ".md", ".sh", ".txt", ".html", ".yml", ".yaml",
    ".json", ".toml", ".ini", ".cfg", ".example", ".env",
    "Dockerfile",
}


# ── HTTP-Hilfsfunktionen ──────────────────────────────────────────────────────

def _request(method: str, path: str, token: str, body=None) -> dict:
    url = API_BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib_error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {path} → HTTP {e.code}: {body_text[:500]}"
        )


def _exists(path: str, token: str) -> bool:
    try:
        _request("GET", path, token)
        return True
    except RuntimeError as e:
        if "HTTP 404" in str(e):
            return False
        raise


# ── Datei-Einsammlung ─────────────────────────────────────────────────────────

def _should_ignore(rel: Path) -> bool:
    for part in rel.parts:
        if part in IGNORE_NAMES:
            return True
    if rel.suffix in IGNORE_SUFFIXES:
        return True
    return False


def collect_files(root: Path) -> list[tuple[str, bytes]]:
    """Gibt Liste von (posix-relativer-Pfad, Bytes) für alle Projektdateien zurück."""
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if _should_ignore(rel):
            continue
        result.append((rel.as_posix(), path.read_bytes()))
    return result


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit(
            "ERROR: GITHUB_TOKEN ist nicht gesetzt.\n"
            "Führe aus:  export GITHUB_TOKEN=ghp_...\n"
            "Dann erneut: python github_push.py"
        )

    print("⬡  GitHub Push via Git Database API")
    print(f"   Repo:     {REPO_NAME}")
    print(f"   Branch:   {BRANCH}")
    print(f"   Message:  {COMMIT_MESSAGE}")
    print()

    # 1. Authentifizierten Nutzer ermitteln
    print("→  Ermittle GitHub-Nutzer…")
    user = _request("GET", "/user", token)
    owner = user["login"]
    print(f"   Owner: {owner}")

    repo_api = f"/repos/{owner}/{REPO_NAME}"

    # 2. Repo anlegen falls nicht vorhanden
    if not _exists(repo_api, token):
        print(f"→  Lege neues Repo {owner}/{REPO_NAME} an…")
        try:
            _request("POST", "/user/repos", token, {
                "name": REPO_NAME,
                "description": (
                    "Self-hosted image analysis via Ollama + LLaVA — "
                    "SQLite gallery, batch upload, re-analyze"
                ),
                "private": False,
                "auto_init": False,
            })
            print("   Repo erstellt.")
        except RuntimeError as e:
            if "HTTP 403" in str(e):
                sys.exit(
                    "\n⚠  Token hat keine Berechtigung, Repos zu erstellen.\n"
                    "   Erstelle das Repo manuell auf GitHub:\n"
                    f"   https://github.com/new  →  Name: {REPO_NAME}  →  Public  →  ohne auto-init\n"
                    "   Dann dieses Skript erneut ausführen."
                )
            raise
    else:
        print(f"→  Repo {owner}/{REPO_NAME} existiert bereits.")

    # 3. Dateien einsammeln
    print(f"→  Sammle Dateien aus {PROJECT_DIR} …")
    files = collect_files(PROJECT_DIR)
    print(f"   {len(files)} Dateien gefunden:")
    for p, content in files:
        print(f"     {p}  ({len(content):,} bytes)")
    print()

    # 4. Blobs erzeugen
    print("→  Erstelle Blobs…")

    def _create_blob(rel_path: str, content: bytes) -> str:
        """Erstellt einen Blob und gibt seinen SHA zurück."""
        is_text = (
            Path(rel_path).suffix in TEXT_SUFFIXES
            or Path(rel_path).name in TEXT_SUFFIXES
        )
        if is_text:
            try:
                blob = _request(
                    "POST", f"{repo_api}/git/blobs", token,
                    {"content": content.decode("utf-8"), "encoding": "utf-8"},
                )
                return blob["sha"]
            except UnicodeDecodeError:
                pass  # Fallback auf Base64
        b64 = base64.b64encode(content).decode()
        blob = _request(
            "POST", f"{repo_api}/git/blobs", token,
            {"content": b64, "encoding": "base64"},
        )
        return blob["sha"]

    blob_shas: dict[str, str] = {}
    for i, (rel_path, content) in enumerate(files):
        try:
            sha = _create_blob(rel_path, content)
        except RuntimeError as e:
            if i == 0 and "HTTP 409" in str(e) and "empty" in str(e).lower():
                # Leeres Repo: Git DB API funktioniert erst nach dem ersten Commit.
                # Initialisieren via Contents API, dann erneut versuchen.
                print("   ↳ Repo ist leer — initialisiere via Contents API…")
                init = _request(
                    "PUT", f"{repo_api}/contents/.gitkeep", token,
                    {
                        "message": "chore: initialize repository",
                        "content": base64.b64encode(b"").decode(),
                    },
                )
                parent_sha = init["commit"]["sha"]
                base_tree_sha = init["commit"]["tree"]["sha"]
                ref_exists = True
                print(f"   ↳ Init-Commit: {parent_sha[:8]}")
                sha = _create_blob(rel_path, content)
            else:
                raise
        blob_shas[rel_path] = sha
        print(f"   blob {sha[:8]}  {rel_path}")

    print()

    # 5. Aktuellen HEAD ermitteln (falls Branch schon existiert)
    ref_path = f"{repo_api}/git/ref/heads/{BRANCH}"
    parent_sha: str | None = None
    base_tree_sha: str | None = None
    ref_exists = _exists(ref_path, token)

    if ref_exists:
        ref = _request("GET", ref_path, token)
        parent_sha = ref["object"]["sha"]
        commit = _request("GET", f"{repo_api}/git/commits/{parent_sha}", token)
        base_tree_sha = commit["tree"]["sha"]
        print(f"→  Bestehender HEAD: {parent_sha[:8]} (tree {base_tree_sha[:8]})")
    else:
        print(f"→  Branch '{BRANCH}' neu — erster Commit.")

    # 6. Tree erstellen
    print("→  Erstelle Tree…")
    tree_items = [
        {
            "path": rel_path,
            "mode": "100755" if rel_path.endswith(".sh") else "100644",
            "type": "blob",
            "sha": sha,
        }
        for rel_path, sha in blob_shas.items()
    ]
    tree_body: dict = {"tree": tree_items}
    if base_tree_sha:
        tree_body["base_tree"] = base_tree_sha

    tree = _request("POST", f"{repo_api}/git/trees", token, tree_body)
    print(f"   Tree SHA: {tree['sha'][:8]}")

    # 7. Commit erstellen
    print("→  Erstelle Commit…")
    commit_body: dict = {
        "message": COMMIT_MESSAGE,
        "tree": tree["sha"],
        "parents": [parent_sha] if parent_sha else [],
    }
    commit = _request("POST", f"{repo_api}/git/commits", token, commit_body)
    print(f"   Commit SHA: {commit['sha'][:8]}")

    # 8. Ref anlegen oder aktualisieren
    print("→  Aktualisiere Ref…")
    if ref_exists:
        _request(
            "PATCH", f"{repo_api}/git/refs/heads/{BRANCH}", token,
            {"sha": commit["sha"], "force": False},
        )
        print(f"   PATCH refs/heads/{BRANCH} → {commit['sha'][:8]}")
    else:
        _request(
            "POST", f"{repo_api}/git/refs", token,
            {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]},
        )
        print(f"   POST  refs/heads/{BRANCH} → {commit['sha'][:8]}")

    print()
    print(f"✓  Fertig!  https://github.com/{owner}/{REPO_NAME}")


if __name__ == "__main__":
    main()
