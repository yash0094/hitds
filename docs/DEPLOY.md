# Deploying HITDS

Pick one. For the judges, **option A is the most reliable**: no hosting bill, no cold starts, the full model
on your laptop, and a public link if judges want to click through.

| | Cost | RAM | Public URL | Notes |
|---|---|---|---|---|
| A. Laptop + Cloudflare quick tunnel | free | 16 GB (yours) | yes, random `*.trycloudflare.com` | laptop must stay on; quick tunnels are for testing, not uptime-guaranteed |
| B. Hugging Face Docker Space | PRO subscription | 16 GB (CPU basic) | `https://<user>-hitds.hf.space` | Docker Spaces need PRO for new personal Spaces (checked Sept 2026); disk not persistent |
| C. Render (Docker) | Standard plan | 2 GB | `https://hitds.onrender.com` | free/Starter (512 MB) is too small unless you build the lite image and slim the model |
| D. College server / any VPS | varies | ≥ 2 GB | your domain | `docker compose up -d` |

Prices and plans change. Check the provider's pricing page before paying.

## Before any hosted option: export the bundle
```
python scripts/06_export_deploy.py --dataset cicids2017
```
This writes `deploy/` with the live model and capped data. If it warns that the model is large (> 400 MB), set
`models.rf.n_estimators: 120` and `models.rf.max_depth: 30`, retrain with `03_train.py --fresh`, and export again.

## A. Laptop + public tunnel
```
winget install --id Cloudflare.cloudflared       # once
.\run_demo.ps1 -Public
```
The tunnel window prints an `https://….trycloudflare.com` URL. Share it with the judges together with the
reviewer login. Change the default passwords first:
`$env:HITDS_USERS="analyst:<pw>,reviewer2:<pw>"`.

## B. Hugging Face Space (Docker)
1. Create a Space with SDK **Docker**. Copy `deploy/hf_space/README.md` over the Space's README (it sets `app_port: 7860`).
2. In the Space's settings, add **secrets** `HITDS_USERS`, `HITDS_SECRET_KEY` and `HITDS_API_KEY`.
3. Push: `Dockerfile`, `requirements-deploy.txt`, `serve.py`, `hitds/`, `app/`, `deploy/` (without `deploy/hf_space`).
   Track the big files with Git LFS: `git lfs track "*.joblib" "*.pkl"`.
4. The build takes a few minutes. Check `https://<space-url>/healthz`.
Data written at runtime (verdicts, retrained models) is lost on restart. Export the audit log from the UI after a session.

## C. Render
Push the repo, including `deploy/`, to GitHub with Git LFS, then in Render create **New → Blueprint** and point
it at `render.yaml`. Set `HITDS_USERS` when prompted. For a 512 MB instance, build the lite image
(`docker build --build-arg INSTALL_TORCH=0`), use `--members rf,svm` and a smaller forest. Test memory
locally first with `docker stats`.

## D. Docker anywhere
```
docker compose up --build -d        # edit the passwords in docker-compose.yml first
```

## Security checklist before sharing a URL
- [ ] `HITDS_USERS` set to non-default passwords, with one login per real person (verdicts are attributed to logins)
- [ ] `HITDS_SECRET_KEY` set to a long random string (otherwise sessions reset on every restart)
- [ ] `HITDS_API_KEY` set only if a collector will post flows; keep it out of Git
- [ ] actions are **simulated** (`executed(simulated)`). Nothing touches a real firewall, so say so in the demo
