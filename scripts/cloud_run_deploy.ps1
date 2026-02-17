param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectId,

  [Parameter(Mandatory = $false)]
  [string]$Region = "europe-west1",

  [Parameter(Mandatory = $false)]
  [string]$ServiceName = "cv-parsing-api",

  [Parameter(Mandatory = $false)]
  [string]$ArtifactRepo = "cv-api-repo",

  [Parameter(Mandatory = $false)]
  [string]$ImageTag = "latest",

  [Parameter(Mandatory = $true)]
  [string]$MistralApiKey,

  [Parameter(Mandatory = $true)]
  [string]$ReplicateApiToken,

  [Parameter(Mandatory = $false)]
  [string]$ApiAuthToken = "",

  [Parameter(Mandatory = $false)]
  [switch]$AllowUnauthenticated
)

$ErrorActionPreference = "Stop"

if ($ProjectId -match "^YOUR_" -or [string]::IsNullOrWhiteSpace($ProjectId)) {
  throw "Invalid ProjectId. Provide a real GCP project id (not placeholder)."
}

function Assert-Command([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Command '$name' is not available. Install it and retry."
  }
}

function Invoke-GCloud {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
  )
  & gcloud @Args
  if ($LASTEXITCODE -ne 0) {
    throw "gcloud failed (exit $LASTEXITCODE): gcloud $($Args -join ' ')"
  }
}

function Invoke-GCloudCapture {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
  )
  $output = & gcloud @Args 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "gcloud failed (exit $LASTEXITCODE): gcloud $($Args -join ' ')`n$output"
  }
  return (($output | Out-String).Trim())
}

function Ensure-GCloudAuth([string]$project) {
  Write-Host "Setting gcloud project: $project" -ForegroundColor Cyan
  Invoke-GCloud config set project $project
  Invoke-GCloud projects describe $project --project $project --format "value(projectNumber)"
}

function Ensure-ServiceEnabled([string]$serviceName, [string]$project) {
  Write-Host "Ensuring API enabled: $serviceName" -ForegroundColor Cyan
  Invoke-GCloud services enable $serviceName --project $project
}

function Ensure-ArtifactRepo([string]$repo, [string]$region, [string]$project) {
  Write-Host "Ensuring Artifact Registry repo exists: $repo ($region)" -ForegroundColor Cyan
  $exists = $false
  try {
    Invoke-GCloud artifacts repositories describe $repo --location $region --project $project
    $exists = $true
  } catch {
    $exists = $false
  }

  if (-not $exists) {
    Invoke-GCloud artifacts repositories create $repo `
      --repository-format docker `
      --location $region `
      --description "CV Parsing API images" `
      --project $project
  }
}

function Upsert-Secret([string]$secretName, [string]$secretValue, [string]$project) {
  Write-Host "Upserting Secret Manager secret: $secretName" -ForegroundColor Cyan
  $exists = $false
  try {
    Invoke-GCloud secrets describe $secretName --project $project
    $exists = $true
  } catch {
    $exists = $false
  }

  if (-not $exists) {
    Invoke-GCloud secrets create $secretName --replication-policy automatic --project $project
  }

  $tmp = [System.IO.Path]::GetTempFileName()
  try {
    Set-Content -Path $tmp -Value $secretValue -NoNewline -Encoding ascii
    Invoke-GCloud secrets versions add $secretName --data-file $tmp --project $project
  } finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  }
}

Assert-Command "gcloud"

Write-Host "Starting Cloud Run deployment..." -ForegroundColor Green
Ensure-GCloudAuth -project $ProjectId

Ensure-ServiceEnabled "run.googleapis.com" -project $ProjectId
Ensure-ServiceEnabled "artifactregistry.googleapis.com" -project $ProjectId
Ensure-ServiceEnabled "cloudbuild.googleapis.com" -project $ProjectId
Ensure-ServiceEnabled "secretmanager.googleapis.com" -project $ProjectId

Ensure-ArtifactRepo -repo $ArtifactRepo -region $Region -project $ProjectId

$imageUri = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepo/$ServiceName`:$ImageTag"
Write-Host "Building image with Cloud Build: $imageUri" -ForegroundColor Cyan
Invoke-GCloud builds submit `
  --config cloudbuild.api.yaml `
  --substitutions "_IMAGE_URI=$imageUri" `
  --project $ProjectId `
  .

$secretMistral = "$ServiceName-mistral-api-key"
$secretReplicate = "$ServiceName-replicate-api-token"
Upsert-Secret -secretName $secretMistral -secretValue $MistralApiKey -project $ProjectId
Upsert-Secret -secretName $secretReplicate -secretValue $ReplicateApiToken -project $ProjectId

$secretMappings = @(
  "MISTRAL_API_KEY=$secretMistral:latest",
  "REPLICATE_API_TOKEN=$secretReplicate:latest"
)

if ($ApiAuthToken -and $ApiAuthToken.Trim()) {
  $secretAuth = "$ServiceName-api-auth-token"
  Upsert-Secret -secretName $secretAuth -secretValue $ApiAuthToken -project $ProjectId
  $secretMappings += "API_AUTH_TOKEN=$secretAuth:latest"
}

$envVars = "APP_ENV=prod,MAX_FILE_SIZE_MB=10,DEFAULT_SLA_SECONDS=45,API_WORKER_THREADS=4"

$deployArgs = @(
  "run", "deploy", $ServiceName,
  "--project", $ProjectId,
  "--image", $imageUri,
  "--region", $Region,
  "--platform", "managed",
  "--port", "8080",
  "--memory", "1Gi",
  "--cpu", "1",
  "--concurrency", "10",
  "--timeout", "300",
  "--max-instances", "10",
  "--set-env-vars", $envVars,
  "--set-secrets", ($secretMappings -join ",")
)

if ($AllowUnauthenticated.IsPresent) {
  $deployArgs += "--allow-unauthenticated"
} else {
  $deployArgs += "--no-allow-unauthenticated"
}

Write-Host "Deploying Cloud Run service..." -ForegroundColor Cyan
Invoke-GCloud @deployArgs

$serviceUrl = Invoke-GCloudCapture run services describe $ServiceName `
  --project $ProjectId `
  --region $Region `
  --format "value(status.url)"

if (-not $serviceUrl) {
  throw "Deployment did not return a service URL."
}

Write-Host "Deployment completed." -ForegroundColor Green
Write-Host "Service URL: $serviceUrl" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: validate with scripts/cloud_run_validate.ps1" -ForegroundColor Yellow
