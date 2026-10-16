$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$files = @(
    "visiontar.db",
    "visiontar.db-shm",
    "visiontar.db-wal",
    "report_lotti.csv"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "Rimosso: $file"
    }
}

python -c "from database import init_db; init_db(); print('Database test ricreato vuoto')"
