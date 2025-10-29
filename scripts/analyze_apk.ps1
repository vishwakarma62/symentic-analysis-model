# APK Analysis Script
param(
    [string]$ApkPath = "..\build\app\outputs\apk\debug\app-debug.apk"
)

$ErrorActionPreference = "Stop"
$TempDir = "temp_apk"
$Username = $env:USERNAME
$NdkPath = "C:\Users\$Username\AppData\Local\Android\Sdk\ndk\27.0.12077973\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-objdump.exe"
$ZipalignPath = "C:\Users\$Username\AppData\Local\Android\Sdk\build-tools\35.0.0\zipalign.exe"

Write-Host "Starting APK Analysis..." -ForegroundColor Cyan

# Create temp directory
if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }
New-Item -ItemType Directory -Path $TempDir | Out-Null

# Extract APK
Write-Host "Extracting APK..." -ForegroundColor Yellow
tar -xf $ApkPath -C $TempDir

# Analyze .so files
$SoFiles = Get-ChildItem "$TempDir\lib\arm64-v8a\*.so" -ErrorAction SilentlyContinue
$ElfAlignmentOk = $true

if ($SoFiles) {
    Write-Host "Analyzing ELF files..." -ForegroundColor Yellow
    foreach ($SoFile in $SoFiles) {
        Write-Host "  Checking: $($SoFile.Name)" -ForegroundColor Gray
        
        $LoadLines = & $NdkPath -p $SoFile.FullName | Select-String "LOAD"
        foreach ($Line in $LoadLines) {
            Write-Host "    $Line" -ForegroundColor White
            
            # Extract align value (simplified pattern matching)
            if ($Line -match "align\s+2\*\*(\d+)") {
                $AlignPower = [int]$matches[1]
                if ($AlignPower -lt 14) {
                    Write-Host "    WARNING: Alignment 2**$AlignPower is less than 2**14" -ForegroundColor Red
                    $ElfAlignmentOk = $false
                }
            }
        }
    }
} else {
    Write-Host "  No .so files found in arm64-v8a" -ForegroundColor Gray
}

# Check APK alignment
Write-Host "Verifying APK alignment..." -ForegroundColor Yellow
$ZipalignResult = & $ZipalignPath -v -c -P 16 4 $ApkPath 2>&1
$ApkAlignmentOk = $LASTEXITCODE -eq 0

Write-Host $ZipalignResult -ForegroundColor White

# Summary
Write-Host "`nSUMMARY:" -ForegroundColor Cyan
if ($ElfAlignmentOk) {
    Write-Host "ELF Alignment: OK" -ForegroundColor Green
} else {
    Write-Host "ELF Alignment: Needs Fix" -ForegroundColor Red
}

if ($ApkAlignmentOk) {
    Write-Host "APK Alignment: OK" -ForegroundColor Green
} else {
    Write-Host "APK Alignment: X Needs Fix" -ForegroundColor Red
}

# Cleanup
Remove-Item $TempDir -Recurse -Force