param(
    [string]$Config = "configs\strong.yaml"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

Write-Host "Running training + testing with config: $Config"
& .\Auto\Scripts\python main.py --config $Config --train --test

Write-Host "Launching Streamlit dashboard..."
Start-Process -FilePath ".\Auto\Scripts\python" -ArgumentList "-m", "streamlit", "run", "streamlit_app.py"
