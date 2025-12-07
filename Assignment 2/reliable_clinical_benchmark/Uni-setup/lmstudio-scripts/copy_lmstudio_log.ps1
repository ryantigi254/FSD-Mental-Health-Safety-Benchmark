$ErrorActionPreference = "Stop"

$src = Join-Path $Env:APPDATA "LM Studio\logs\main.log"

if (-not (Test-Path $src)) {
    Write-Host "LM Studio log not found at $src"
    exit 1
}

$destDir = Join-Path $PSScriptRoot "..\results\logs"
New-Item -ItemType Directory -Path $destDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dest = Join-Path $destDir ("lmstudio-main-{0}.log" -f $timestamp)

Copy-Item $src $dest -Force
Write-Host "Copied LM Studio log to $dest"

