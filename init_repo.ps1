$ErrorActionPreference = "Stop"

Write-Host "Preparing repository structure..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path "car_vision_project\data" | Out-Null
New-Item -ItemType Directory -Force -Path "car_vision_project\artifacts" | Out-Null

if (-not (Test-Path "car_vision_project\data\.gitkeep")) {
    New-Item -ItemType File -Path "car_vision_project\data\.gitkeep" | Out-Null
}

if (-not (Test-Path "car_vision_project\artifacts\.gitkeep")) {
    New-Item -ItemType File -Path "car_vision_project\artifacts\.gitkeep" | Out-Null
}

Write-Host "Initializing git repository..." -ForegroundColor Cyan
git init

Write-Host "Switching to main branch..." -ForegroundColor Cyan
git branch -M main

Write-Host "Staging files..." -ForegroundColor Cyan
git add .

Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "feat: initial production-ready car vision system"

Write-Host ""
Write-Host "Repository initialized successfully." -ForegroundColor Green
Write-Host "Now run the following commands to connect GitHub and push:" -ForegroundColor Yellow
Write-Host ""
Write-Host "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
Write-Host "git push -u origin main"
