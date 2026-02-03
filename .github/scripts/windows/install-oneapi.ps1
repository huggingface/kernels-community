param(
    [Parameter(Mandatory=$true)]
    [string]$OneApiVersion,
    
    [Parameter(Mandatory=$true)]
    [string]$OneApiUrl
)

$url = $OneApiUrl
$installer = "$env:TEMP\oneapi_installer.exe"

# Remove old installer if exists to avoid corruption
if (Test-Path $installer) {
    Write-Host "Removing old installer..." -ForegroundColor Yellow
    Remove-Item $installer -Force
}

# Download installer using WebClient for large files
Write-Host "Downloading Intel oneAPI Base Toolkit $OneApiVersion..."
Write-Host "This may take several minutes, please wait..."

try {
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($url, $installer)
    $webClient.Dispose()
} catch {
    throw "Download failed: $_"
}

# Check download success
if (Test-Path $installer) {
    $sizeMB = [math]::Round((Get-Item $installer).Length / 1MB, 2)
    Write-Host "Installer downloaded successfully: $installer" -ForegroundColor Green
    Write-Host "File size: $sizeMB MB" -ForegroundColor Green
} else {
    throw "Download failed"
}

# Install Intel oneAPI Base Toolkit
Write-Host "Installing Intel oneAPI Base Toolkit..."
Write-Host "Note: This may take 10-20 minutes depending on your system..." -ForegroundColor Yellow

# Check if oneAPI is already installed
$oneAPIPath = "C:\Program Files (x86)\Intel\oneAPI"
if (Test-Path $oneAPIPath) {
    Write-Host "Warning: Existing oneAPI installation detected at: $oneAPIPath" -ForegroundColor Yellow
    Write-Host "The installer will attempt to upgrade/modify the existing installation." -ForegroundColor Yellow
}

$installArgs = @("--silent", "--eula", "accept")
$process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru -NoNewWindow

# Check installation result
if ($process.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "Installation failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    Write-Host ""
    throw "Intel oneAPI installation failed with exit code $($process.ExitCode)"
}

# Clean up installer file
Remove-Item $installer -ErrorAction SilentlyContinue
Write-Host "Intel oneAPI Base Toolkit installed successfully" -ForegroundColor Green

# Verify installation
Write-Host ""
Write-Host "Verifying Intel oneAPI installation..." -ForegroundColor Cyan

# Check installation directory
if (Test-Path $oneAPIPath) {
    Write-Host ""
    Write-Host "oneAPI installation directory found: $oneAPIPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installed components:"
    Get-ChildItem $oneAPIPath | Select-Object Name | Format-Table -AutoSize
} else {
    throw "oneAPI installation directory not found at $oneAPIPath"
}
