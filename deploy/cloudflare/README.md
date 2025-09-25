# CI/CD via Cloudflare + GitHub

Goal: On every push to main, automatically redeploy the app behind your domain using Cloudflare Tunnel.

Two models:
- Self-hosted runner on the same machine where the app runs (simplest)
- Remote build + SSH to the host (works too, but needs SSH secrets)

This guide covers the self-hosted runner approach.

## Overview
- You run a GitHub Actions self-hosted runner on your GPU host.
- A workflow pulls latest main, installs deps, restarts services for backend and frontend, and restarts cloudflared.
- Cloudflare Tunnel maps:
  - ui.example.com -> http://127.0.0.1:7860 (Gradio)
  - api.example.com -> http://127.0.0.1:8000 (FastAPI)

## Setup Steps

1) Create Cloudflare Tunnel and DNS
- Install cloudflared and login:
  cloudflared tunnel login
- Create a tunnel named gpt-oss:
  cloudflared tunnel create gpt-oss
- Create ~/.cloudflared/config.yml like:
  tunnel: gpt-oss
  credentials-file: /home/USER/.cloudflared/gpt-oss.json
  ingress:
    - hostname: ui.example.com
      service: http://127.0.0.1:7860
    - hostname: api.example.com
      service: http://127.0.0.1:8000
    - service: http_status:404
- In Cloudflare DNS, add CNAME records for ui.example.com and api.example.com pointing to the tunnel.

2) Systemd services (on the host)
- Create services for backend, frontend, and cloudflared. Templates are in this folder:
  - gpt-oss-backend.service
  - gpt-oss-frontend.service
  - cloudflared.service (optional if not already installed by package)

3) GitHub Actions self-hosted runner
- On your repo: Settings → Actions → Runners → New self-hosted runner
- Install the runner on the host and connect it to the repo.

4) Deploy script and GitHub Actions workflow
- The deploy.sh script restarts services after pulling the latest code.
- The workflow .github/workflows/deploy.yml runs on push to main, on the self-hosted runner.

## Commands
- Start services:
  sudo systemctl daemon-reload
  sudo systemctl enable gpt-oss-backend gpt-oss-frontend cloudflared
  sudo systemctl start gpt-oss-backend gpt-oss-frontend cloudflared
- Check status:
  systemctl status gpt-oss-backend
  systemctl status gpt-oss-frontend
  systemctl status cloudflared

## Notes
- For tunnels or proxies, prefer non-streaming in UI (we defaulted that).
- If you change ports, update both systemd units and cloudflared config.
