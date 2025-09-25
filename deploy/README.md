# Deploying GPT-OSS with a custom domain

This guide shows two practical options to put your local app behind a domain with HTTPS:

- Option A: Cloudflared Tunnel + Cloudflare DNS (free, simple)
- Option B: Nginx reverse proxy on a VPS + your DNS

Both approaches avoid VS Code dev-tunnel limitations (CORS and long-polling timeouts) and work well with Gradio/FastAPI.

## Option A: Cloudflared Tunnel (recommended)

What you get:
- A stable HTTPS URL on your domain (e.g., ui.example.com)
- No inbound ports opened on your machine
- Automatic certs; great for development and small teams

### Prerequisites
- A domain managed in Cloudflare
- Install cloudflared on the machine running GPT-OSS

### Steps
1. Login and create a named tunnel:
   cloudflared tunnel login
   cloudflared tunnel create gpt-oss

2. Route your subdomains to the tunnel in Cloudflare DNS:
   - ui.example.com -> Tunnel (HTTP) to http://127.0.0.1:7860
   - api.example.com -> Tunnel (HTTP) to http://127.0.0.1:8000

3. Create a config file at ~/.cloudflared/config.yml:
   tunnel: gpt-oss
   credentials-file: /home/USER/.cloudflared/gpt-oss.json
   ingress:
     - hostname: ui.example.com
       service: http://127.0.0.1:7860
     - hostname: api.example.com
       service: http://127.0.0.1:8000
     - service: http_status:404

4. Run the tunnel:
   cloudflared tunnel run gpt-oss

5. In the UI, set Backend URL to https://api.example.com. Keep Stream output off unless you test it stable.

Notes:
- If you only want the frontend public, skip api.example.com and keep the backend on 127.0.0.1. The frontend will reach it locally.
- You can restrict access with Cloudflare Access (SSO) later.

## Option B: Nginx Reverse Proxy on a VPS

What you get:
- Full control on a public VPS (e.g., Lightsail, Droplet, EC2)
- Hostname with HTTPS via Let’s Encrypt

### Prerequisites
- VPS with a public IP
- DNS A records:
  - ui.example.com -> VPS IP
  - api.example.com -> VPS IP

### Steps
1. On the VPS, install nginx and certbot.
2. Create reverse proxy sites:
   - ui.example.com proxy_pass to your workstation’s tunnel or VPN URL for port 7860
   - api.example.com proxy_pass to your workstation’s tunnel or VPN URL for port 8000
3. Obtain certificates:
   sudo certbot --nginx -d ui.example.com -d api.example.com
4. Update the app’s Backend URL to https://api.example.com in the UI.

Notes:
- This is better if you plan production traffic or multi-user access.
- You may run both frontend and backend on the VPS itself to simplify routing.

## Gradio and Backend Settings
- Frontend UI (Gradio):
  - Prefer non-streaming (Stream output off) through proxies
  - Disable auto-refresh timers in high-latency networks
- Backend (FastAPI):
  - You can bind to 0.0.0.0 to be accessible from the proxy
  - CORS is already permissive in this repo for dev setups

## Troubleshooting
- 504/timeout: increase proxy timeouts, prefer non-streaming, keep queue disabled
- CORS errors: confirm DNS points correctly and proxy preserves headers
- Web manifest errors: our UI removes manifest tags to prevent proxy rewrite issues

## Next steps
If you want, I can generate a sample cloudflared config and a small script to start both the UI and the tunnel so you can be live on your domain in one command.
