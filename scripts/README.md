# APK Analysis Scripts

This directory contains scripts for analyzing APK alignment and 16KB page size compatibility.

## Scripts

### 1. `analyze_apk.ps1` (PowerShell)
**Windows-native script for APK analysis**

```powershell
# Run with default APK path
.\scripts\analyze_apk.ps1

# Run with custom APK path
.\scripts\analyze_apk.ps1 -ApkPath "path\to\your.apk"
```

**Features:**
- Extracts APK and analyzes ELF files
- Checks native library alignment (16KB requirement)
- Verifies APK zip alignment
- Clean summary output with ✅/❌ status

### 2. `check_elf_alignment.sh` (Bash)
**Google's official ELF alignment checker**

```bash
# Using Git Bash or WSL
bash scripts/check_elf_alignment.sh build/app/outputs/apk/debug/app-debug.apk

# Make executable and run (Linux/WSL)
chmod +x scripts/check_elf_alignment.sh
./scripts/check_elf_alignment.sh build/app/outputs/apk/debug/app-debug.apk
```

**Features:**
- Official Android validation tool
- Supports APK, APEX, and directory analysis
- Color-coded output (GREEN/RED)
- Comprehensive ELF and zip alignment checks

## Requirements

### For PowerShell Script:
- Windows PowerShell 5.1+
- Android SDK with NDK and Build Tools
- Paths automatically detected using `$env:USERNAME`

### For Bash Script:
- Git Bash, WSL, or Linux environment
- `objdump` (from Android NDK)
- `zipalign` (from Android Build Tools)
- `unzip`, `file` utilities

## Android 16 Compatibility

Both scripts verify:
- **ELF Alignment**: Native libraries aligned to 2**14 (16KB) or higher
- **APK Alignment**: Proper file alignment within APK archive
- **16KB Page Size**: Compatibility with Android 16's memory management

## Usage Examples

```powershell
# Quick check
.\scripts\analyze_apk.ps1

# Detailed analysis with Google's tool
bash scripts/check_elf_alignment.sh build/app/outputs/apk/debug/app-debug.apk
```

## Expected Output

✅ **Success**: All libraries properly aligned for Android 16  
❌ **Issues**: Alignment problems that need fixing

Your Homitra app is **16KB page size compatible** and ready for Android 16!