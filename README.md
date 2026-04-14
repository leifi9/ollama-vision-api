# Ollama Vision API – Lokale Bilderkennung mit LLaVA

Self-hosted Backend-Service für Bildanalyse via Ollama + LLaVA.
Erkennt Objekte auf Fotos und gibt strukturierte JSON-Daten zurück
(Name, Keywords, Beschreibung, STL-Suchbegriff).

## Features

- **100% lokal** – keine Cloud, keine API-Keys, volle Datenkontrolle
- **FastAPI** Backend mit Swagger-Doku (`/docs`)
- **LLaVA Vision Model** via Ollama REST API
- **Upload + Analyse** in einem Request
- **Galerie** mit persistentem SQLite-Katalog, Volltextsuche und Re-Analyse
- **Batch-Upload** mit Server-Sent Events (SSE) Fortschrittsanzeige
- **Web-UI** direkt im Browser (Tab: Analyze | Gallery)
- **Docker-ready** (optional)

## Voraussetzungen

- Python 3.10+
- [Ollama](https://ollama.com) installiert und laufend
- Min. 8 GB VRAM (GPU) oder 16 GB RAM (CPU, deutlich langsamer)

## Schnellstart

```bash
# 1. Ollama starten & Modell laden
ollama serve &
ollama pull llava:7b

# 2. Python-Umgebung einrichten
cd ollama-vision-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Konfiguration (optional – Defaults funktionieren)
cp .env.example .env

# 4. Server starten
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Browser öffnen
# API Docs:  http://localhost:8000/docs
# Web UI:    http://localhost:8000/
```

## Quick Demo

Sobald der Server läuft, teste den kompletten Batch-→-Galerie-Flow mit einem Befehl:

```bash
cd ~/projects/ollama-vision-api
./examples/demo.sh
```

Das Skript sendet 5 Referenz-Bilder als Batch, liest den SSE-Stream und zeigt
den Fortschritt pro Bild. Danach öffne `http://localhost:8000/#gallery`.

---

## API Endpunkte

### Vision (Einzelbild)

| Methode | Pfad | Beschreibung |
|---------|------|--------------|
| POST | `/api/analyze` | Bild hochladen und analysieren |
| GET | `/api/health` | Health-Check (Ollama-Verbindung) |
| GET | `/api/models` | Verfügbare Vision-Modelle auflisten |

### Gallery

| Methode | Pfad | Beschreibung |
|---------|------|--------------|
| GET | `/api/gallery` | Alle Bilder mit letzter Analyse auflisten |
| GET | `/api/gallery/search?q=...` | Volltextsuche über alle Analysen |
| GET | `/api/gallery/{id}` | Einzelnes Bild mit allen Analysen |
| POST | `/api/gallery/{id}/reanalyze` | Bild erneut analysieren |
| DELETE | `/api/gallery/{id}` | Bild und alle Analysen löschen |
| POST | `/api/gallery/batch` | Mehrere Bilder hochladen (SSE-Stream) |
| GET | `/api/images/{filename}` | Gespeichertes Bild ausliefern |
| GET | `/` | Web-UI |

### curl-Beispiele

```bash
# Einzelbild analysieren und in Galerie speichern
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@foto.jpg" \
  -F "prompt=Describe this object for 3D printing"

# Galerie auflisten (Seite 1, 20 Einträge)
curl http://localhost:8000/api/gallery?page=1&per_page=20

# Volltextsuche
curl "http://localhost:8000/api/gallery/search?q=gear"

# Einzelbild mit allen Analysen
curl http://localhost:8000/api/gallery/42

# Re-Analyse mit Custom-Prompt
curl -X POST http://localhost:8000/api/gallery/42/reanalyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe this object in detail", "model": null}'

# Bild löschen
curl -X DELETE http://localhost:8000/api/gallery/42

# Batch-Upload (SSE-Stream, -N = kein Buffering)
curl -N -X POST http://localhost:8000/api/gallery/batch \
  -F "files=@foto1.jpg" \
  -F "files=@foto2.jpg"
```

---

## Gallery

Die Galerie ist ein persistenter SQLite-Katalog aller hochgeladenen Bilder und ihrer Analysen.

**Was sie kann:**
- Bilder persistent speichern (Disk) mit Metadaten in SQLite
- Mehrere Analysen pro Bild speichern (verschiedene Prompts / Modelle)
- Alle Analysen eines Bilds im Detail-View **nebeneinander** vergleichen
- Volltextsuche (FTS5) über Name, Keywords, Beschreibung und STL-Suchbegriff
- Batch-Upload mit Echtzeit-Fortschritt via SSE

**Wo die DB liegt:** `~/.ollama-vision/gallery.db` (Standard)

**Backup:** Einfach die Datei kopieren — `cp ~/.ollama-vision/gallery.db ~/backup/`

**Reset:** Datei löschen, neu starten — `rm ~/.ollama-vision/gallery.db`

**Bilder:** `~/.ollama-vision/images/` (Standard, je ein UUID-Filename)

Beide Pfade sind via `.env` überschreibbar (`DATABASE_PATH`, `PERSISTENT_UPLOAD_DIR`).

---

## Konfiguration (.env)

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama Server URL |
| VISION_MODEL | llava:7b | Modellname |
| MAX_FILE_SIZE_MB | 20 | Max Upload-Größe pro Datei |
| UPLOAD_DIR | ./uploads | Temp-Upload-Verzeichnis |
| DATABASE_PATH | ~/.ollama-vision/gallery.db | SQLite-Datenbankdatei |
| PERSISTENT_UPLOAD_DIR | ~/.ollama-vision/images | Dauerhafter Bildspeicher |
| MAX_BATCH_SIZE | 20 | Max Dateien pro Batch-Request |

## Projektstruktur

```
ollama-vision-api/
├── backend/
│   ├── main.py              # FastAPI App + Startup (init_db)
│   ├── config.py            # Settings via pydantic-settings
│   ├── database.py          # SQLite: Schema-Migrationen + CRUD + FTS5-Suche
│   ├── routers/
│   │   ├── analyze.py       # POST /api/analyze, GET /api/health, /api/models
│   │   └── gallery.py       # Gallery + Batch + Re-Analyze Endpoints
│   ├── services/
│   │   ├── vision_service.py # Ollama/LLaVA Integration
│   │   └── gallery_service.py # Bild auf Disk schreiben + DB-Row anlegen
│   └── static/
│       └── index.html        # Web-UI (Analyze | Gallery Tabs)
├── examples/
│   ├── demo-images/          # 5 Referenz-Bilder für demo.sh
│   └── demo.sh               # Batch-Demo-Script (curl + SSE)
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```
