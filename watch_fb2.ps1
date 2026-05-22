$ErrorActionPreference = "SilentlyContinue"
$root = "C:\Users\1\Desktop\НИКОЛАЕВ"
$inbox = Join-Path $root "backend\media\books\inbox"
New-Item -ItemType Directory -Force -Path $inbox | Out-Null
while ($true) {
  Get-ChildItem -Path $root -File | Where-Object { $_.Extension -in @('.fb2', '.pdf') } | ForEach-Object {
    $target = Join-Path $inbox $_.Name
    if (-not (Test-Path $target)) {
      Move-Item -LiteralPath $_.FullName -Destination $target -Force
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Add-Content -Path (Join-Path $root "run_logs\fb2_watcher.log") -Value "[$timestamp] moved $($_.Name) -> $target"
    }
  }
  Start-Sleep -Seconds 2
}
