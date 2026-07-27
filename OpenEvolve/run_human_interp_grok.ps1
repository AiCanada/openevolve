# Standalone OpenEvolve run driven by Grok (xAI) only - isolated from Claude.
#
# Usage:
#   cd "A:\Git Hub\openevolve\OpenEvolve"
#   $env:XAI_API_KEY = "xai-..."
#   powershell -ExecutionPolicy Bypass -File .\run_human_interp_grok.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_human_interp_grok.ps1 -Iterations 20
#   powershell -ExecutionPolicy Bypass -File .\run_human_interp_grok.ps1 -DryEvalOnly

param(
    [int]$Iterations = 500,
    [string]$Out = "examples/human_interp_path/openevolve_output_grok",
    [string]$Device = "cuda",
    [string]$Model = "",
    [switch]$DryEvalOnly,
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONPATH = $Root
$env:HUMAN_EVOLVE_DEVICE = $Device
$env:PYTHONIOENCODING = "utf-8"

# Do not inherit Claude usage for this process
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue

if (-not $env:XAI_API_KEY) {
    if ($env:OPENAI_API_KEY -and $env:OPENAI_API_KEY.StartsWith("xai-")) {
        $env:XAI_API_KEY = $env:OPENAI_API_KEY
        Write-Host "Using OPENAI_API_KEY as XAI_API_KEY (xai- prefix detected)" -ForegroundColor Yellow
    }
}

$keyStatus = if ($env:XAI_API_KEY) { "SET (len=$($env:XAI_API_KEY.Length))" } else { "MISSING" }

Write-Host "=== OpenEvolve (Grok/xAI only - separate from Claude) ===" -ForegroundColor Cyan
Write-Host "Root:     $Root"
Write-Host "Device:   $Device"
Write-Host "Out:      $Out"
Write-Host "Iters:    $Iterations"
Write-Host "XAI key:  $keyStatus"

if (-not (Test-Path "examples/human_interp_path/windows.npz")) {
    throw "Missing windows.npz under examples/human_interp_path/"
}
if (-not (Test-Path "examples/human_interp_path/config_grok.yaml")) {
    throw "Missing config_grok.yaml"
}

Write-Host ""
Write-Host "=== Dry eval seed on $Device ===" -ForegroundColor Green
& py -3 examples/human_interp_path/evaluator.py examples/human_interp_path/initial_program.py evolve
if ($LASTEXITCODE -ne 0) { throw "Seed evaluator failed exit=$LASTEXITCODE" }

if ($DryEvalOnly) {
    Write-Host "DryEvalOnly - not starting evolution." -ForegroundColor Yellow
    exit 0
}

if (-not $env:XAI_API_KEY) {
    throw @"
XAI_API_KEY is not set. Create a key at https://console.x.ai/ then:

  `$env:XAI_API_KEY = 'xai-...'
  powershell -ExecutionPolicy Bypass -File .\run_human_interp_grok.ps1

This run is intentionally isolated from Claude Code.
"@
}

# Prefer v6 dual seed (folding-focused champ) when present
$seed = "examples/human_interp_path/initial_program.py"
if (Test-Path "examples/human_interp_path/initial_program_v6_seed.py") {
    $seed = "examples/human_interp_path/initial_program_v6_seed.py"
    Write-Host "Seed:     $seed (v6 dual folding champ)" -ForegroundColor Yellow
}

$pyArgs = @(
    "-3",
    "openevolve-run.py",
    $seed,
    "examples/human_interp_path/evaluator.py",
    "--config", "examples/human_interp_path/config_grok.yaml",
    "--iterations", "$Iterations",
    "--output", $Out
)
if ($Model -ne "") {
    $pyArgs += @("--primary-model", $Model)
}
if ($Checkpoint -ne "") {
    $pyArgs += @("--checkpoint", $Checkpoint)
}

Write-Host ""
Write-Host "=== Evolution (Grok) ===" -ForegroundColor Green
Write-Host ("py " + ($pyArgs -join " "))
& py @pyArgs
$code = $LASTEXITCODE
if ($code -ne 0) { throw "openevolve-run.py exited $code" }
Write-Host "Done. Best under $Out\best\" -ForegroundColor Green
