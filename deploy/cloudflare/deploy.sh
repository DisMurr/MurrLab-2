#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"/../..

# Pull latest
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git fetch origin
  git reset --hard origin/main
fi

# Create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Restart services
sudo systemctl daemon-reload
sudo systemctl restart gpt-oss-backend || true
sudo systemctl restart gpt-oss-frontend || true
sudo systemctl restart cloudflared || true

echo "Deploy complete."