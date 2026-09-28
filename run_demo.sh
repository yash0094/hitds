#!/usr/bin/env bash
# HITDS demo launcher (macOS / Linux).   ./run_demo.sh [dataset] [--public]
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HITDS_DATASET="${1:-cicids2017}"
export HITDS_USERS="${HITDS_USERS:-analyst:hitds-demo,reviewer2:hitds-review}"
export HITDS_SECRET_KEY="${HITDS_SECRET_KEY:-$(python -c 'import secrets;print(secrets.token_hex(32))')}"
export PORT="${PORT:-5000}" HOST=127.0.0.1 HITDS_RESET=1
if [ "${2:-}" = "--public" ]; then
  command -v cloudflared >/dev/null || { echo "install cloudflared first"; exit 1; }
  cloudflared tunnel --url "http://127.0.0.1:${PORT}" &
fi
echo "Logins: $HITDS_USERS"
python serve.py
