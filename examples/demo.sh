#!/usr/bin/env bash
# demo.sh — Quick Demo: Batch-Analyse der 5 Referenz-Bilder
#
# Voraussetzungen:
#   - Server läuft unter http://localhost:8000
#   - Ollama läuft mit llava:7b geladen
#
# Starten: cd ~/projects/ollama-vision-api && ./examples/demo.sh

set -euo pipefail

BASE_URL="${OLLAMA_VISION_URL:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$SCRIPT_DIR/demo-images"

# ── Farben ────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
DIM='\033[0;90m'
CYAN='\033[0;36m'
RESET='\033[0m'

log()  { echo -e "${DIM}$*${RESET}"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
err()  { echo -e "${RED}✕${RESET} $*" >&2; }
head() { echo -e "${CYAN}$*${RESET}"; }

# ── Health Check ───────────────────────────────────────
head "⬡ Ollama Vision API — Demo"
echo ""
log "Prüfe Server ($BASE_URL)…"

if ! curl -sf "$BASE_URL/api/health" -o /dev/null; then
  err "Server nicht erreichbar. Starte ihn mit:"
  err "  cd ~/projects/ollama-vision-api"
  err "  python -m uvicorn backend.main:app --reload --port 8000"
  exit 1
fi

health=$(curl -sf "$BASE_URL/api/health")
status=$(echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
model=$(echo  "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('vision_model','?'))")

if [ "$status" != "ok" ]; then
  err "Ollama-Status: $status (Modell: $model)"
  err "Führe aus: ollama pull $model"
  exit 1
fi
ok "Server verbunden · Modell: $model"
echo ""

# ── Bilder prüfen ──────────────────────────────────────
images=("$IMAGES_DIR"/gear.png "$IMAGES_DIR"/phone_stand.png \
        "$IMAGES_DIR"/hook.png  "$IMAGES_DIR"/knob.png \
        "$IMAGES_DIR"/bracket.png)

for img in "${images[@]}"; do
  if [ ! -f "$img" ]; then
    err "Demo-Bild fehlt: $img"
    exit 1
  fi
done

log "Sende ${#images[@]} Bilder an POST $BASE_URL/api/gallery/batch …"
echo ""

# ── Batch POST + SSE stream lesen ─────────────────────
succeeded=0
failed=0

# Baue curl -F Argumente auf
form_args=()
for img in "${images[@]}"; do
  form_args+=(-F "files=@$img")
done

curl -sfN \
  "${form_args[@]}" \
  "$BASE_URL/api/gallery/batch" \
| while IFS= read -r line; do
    # SSE Zeilen haben das Format: "data: {...}"
    if [[ "$line" == data:* ]]; then
      payload="${line#data: }"
      type=$(echo "$payload" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('type',''))" 2>/dev/null || true)

      case "$type" in
        batch_start)
          total=$(echo "$payload" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))")
          log "Batch gestartet: $total Bilder"
          ;;
        image_done)
          name=$(echo "$payload" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('name') or d.get('filename','?'))")
          fname=$(echo "$payload" | python3 -c "import sys,json; print(json.load(sys.stdin).get('filename','?'))")
          echo -e "  ${GREEN}✓${RESET} $fname → ${CYAN}$name${RESET}"
          ;;
        image_error)
          fname=$(echo "$payload" | python3 -c "import sys,json; print(json.load(sys.stdin).get('filename','?'))")
          msg=$(echo "$payload" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('message','?'))")
          echo -e "  ${RED}✕${RESET} $fname: $msg"
          ;;
        batch_aborted)
          msg=$(echo "$payload" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('reason',{}).get('message','?'))")
          err "Batch abgebrochen: $msg"
          exit 1
          ;;
        batch_complete)
          suc=$(echo "$payload" | python3 -c "import sys,json; print(json.load(sys.stdin).get('succeeded',0))")
          fai=$(echo "$payload" | python3 -c "import sys,json; print(json.load(sys.stdin).get('failed',0))")
          echo ""
          ok "Batch abgeschlossen: $suc analysiert · $fai fehlgeschlagen"
          if [ "$fai" -gt 0 ]; then exit 1; fi
          ;;
      esac
    fi
  done

echo ""
echo -e "Galerie öffnen: ${CYAN}$BASE_URL/#gallery${RESET}"
