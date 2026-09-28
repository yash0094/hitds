# HITDS demo launcher (Windows PowerShell).
#   .\run_demo.ps1                      # local only, http://127.0.0.1:5000
#   .\run_demo.ps1 -Public              # also opens a free Cloudflare quick tunnel -> public https URL for judges
#   .\run_demo.ps1 -Dataset synthetic   # offline smoke demo
param(
  [string]$Dataset = "cicids2017",
  [int]$Port = 5000,
  [switch]$Public,
  [switch]$KeepData
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .venv\Scripts\Activate.ps1 }

if (-not $env:HITDS_USERS)      { $env:HITDS_USERS = "analyst:hitds-demo,reviewer2:hitds-review" }
if (-not $env:HITDS_SECRET_KEY) { $env:HITDS_SECRET_KEY = [guid]::NewGuid().ToString() + [guid]::NewGuid().ToString() }
$env:HITDS_DATASET = $Dataset
$env:PORT = "$Port"
$env:HOST = "127.0.0.1"
if (-not $KeepData) { $env:HITDS_RESET = "1" }

if ($Public) {
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared not found. Install once with:  winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
    exit 1
  }
  Write-Host "Starting a Cloudflare quick tunnel - the public https URL appears in the new window." -ForegroundColor Cyan
  Start-Process cloudflared -ArgumentList "tunnel --url http://127.0.0.1:$Port"
}
Write-Host "Logins: $env:HITDS_USERS" -ForegroundColor Cyan
python serve.py
