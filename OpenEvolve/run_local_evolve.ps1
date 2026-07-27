# Local GPU creative param evolution only — no API keys, no max-runtime kill from this script.
#
# Usage:
#   cd "A:\Git Hub\openevolve\OpenEvolve"
#   powershell -ExecutionPolicy Bypass -File .\run_local_evolve.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_local_evolve.ps1 -Iterations 500 -Device cuda
#   powershell -ExecutionPolicy Bypass -File .\run_local_evolve.ps1 -Iterations 100 -Device cuda

param(
    [int]$Iterations = 500,
    [string]$Device = "cuda",
    [int]$Seed = 42,
    [int]$ScreenWindows = 30,
    [int]$Islands = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONPATH = $Root
$env:HUMAN_EVOLVE_DEVICE = $Device
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# Explicitly local: do not use remote LLM APIs
Remove-Item Env:XAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

Write-Host "=== OpenEvolve local GPU CREATIVE v7 (no API) ===" -ForegroundColor Cyan
Write-Host "Root:  $Root"
Write-Host "Device: $Device"
Write-Host "Iters:  $Iterations"
Write-Host "Islands: $Islands"
Write-Host "Script: run_param_evolve_v7.py"
Write-Host "Note:   archive + Pareto + crossover + structural modes; runs until finished"

& py -3 -u run_param_evolve_v7.py `
    --iterations $Iterations `
    --device $Device `
    --screen-windows $ScreenWindows `
    --seed $Seed `
    --islands $Islands

if ($LASTEXITCODE -ne 0) { throw "run_param_evolve_v7.py exited $LASTEXITCODE" }
Write-Host "Done. Best under examples\human_interp_path\openevolve_output_grok\best\" -ForegroundColor Green
Write-Host "Summary: examples\human_interp_path\openevolve_output_grok\param_evolve_v7_summary.json"
