param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId ="project-a89de931-8ab3-4f90-bbb",
    [string]$Region = "us-central1",
    [string]$ServiceName = "desert-vision",
    [string]$AppConfig = "configs/deploy.yaml",
    [string]$AppCheckpointPath = "models/best.pth",
    [string]$AppCheckpointUrl = "",
    [string]$Memory = "4Gi",
    [string]$Cpu = "2",
    [int]$Timeout = 600,
    [int]$MinInstances = 0,
    [int]$MaxInstances = 2
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI is not installed or not available in PATH."
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

Write-Host "Setting active GCP project: $ProjectId"
gcloud config set project $ProjectId

$subs = @(
    "_SERVICE_NAME=$ServiceName"
    "_REGION=$Region"
    "_APP_CONFIG=$AppConfig"
    "_APP_CHECKPOINT_PATH=$AppCheckpointPath"
    "_APP_CHECKPOINT_URL=$AppCheckpointUrl"
    "_MEMORY=$Memory"
    "_CPU=$Cpu"
    "_TIMEOUT=$Timeout"
    "_MIN_INSTANCES=$MinInstances"
    "_MAX_INSTANCES=$MaxInstances"
) -join ","

Write-Host "Submitting Cloud Build for Cloud Run deployment..."
gcloud builds submit --config cloudbuild.yaml --substitutions $subs

Write-Host "Deployment submitted successfully."
