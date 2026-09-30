# Build a standalone ChemInsight3D bundle for Windows with PyInstaller.
#
# The bundle contains its own Python, every package at the version pinned in
# requirements.txt, and the C++ extensions, so nothing installed on the target
# machine can change or break it. It runs on 64-bit Windows.
#
# Needs: Python 3.12 (python.org install, used through the py launcher) and the
# Visual Studio Build Tools with the C++ workload.
#
# Usage (from PowerShell):
#   powershell -ExecutionPolicy Bypass -File build_bundle.ps1
#   powershell -ExecutionPolicy Bypass -File build_bundle.ps1 -Zip
#   powershell -ExecutionPolicy Bypass -File build_bundle.ps1 -Python C:\path	o\python.exe
#
# The work is done in a local folder (default %LOCALAPPDATA%\ChemInsight3D-build)
# because building from a \\wsl.localhost path is slow and unreliable.
# Output: <WorkDir>\dist\ChemInsight3D\ChemInsight3D.exe  (and ChemInsight3D-windows.zip with -Zip)

param(
    [string]$PythonVersion = "3.12",
    [string]$Python = "",   # python.exe to build with; default: py -$PythonVersion
    [string]$WorkDir = "$env:LOCALAPPDATA\ChemInsight3D-build",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
$BuildTools = @("pyinstaller==6.22.3", "pyinstaller-hooks-contrib==2026.7", "altgraph==0.17.5")
$Extensions = @("overlap_matrix", "electron_density_opt_omp", "localization_native")

$Venv = Join-Path $WorkDir "venv"
$Stage = Join-Path $WorkDir "stage"
$Dist = Join-Path $WorkDir "dist"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Invoke-Checked {
    # Run a native command and stop if it fails.
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

New-Item -ItemType Directory -Force $WorkDir | Out-Null
$env:PYTHONNOUSERSITE = "1"   # ignore packages in the user site folder

Write-Host "==> Build environment (Python $PythonVersion)"
if (-not (Test-Path $VenvPython)) {
    if ($Python) {
        Invoke-Checked $Python @("-m", "venv", $Venv)
    } else {
        Invoke-Checked "py" @("-$PythonVersion", "-m", "venv", $Venv)
    }
}
Invoke-Checked $VenvPython @("-m", "pip", "install", "-q", "--upgrade", "pip")
Invoke-Checked $VenvPython (@("-m", "pip", "install", "-q", "-r", (Join-Path $Repo "requirements.txt")) + $BuildTools)

Write-Host "==> Staging sources"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force $Stage | Out-Null
Copy-Item (Join-Path $Repo "*.py") $Stage
Copy-Item (Join-Path $Repo "*.cpp") $Stage

Write-Host "==> Compiling C++ extensions with MSVC"
Invoke-Checked $VenvPython @((Join-Path $Stage "build_native_extensions.py"))

$ExtSuffix = & $VenvPython -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
$AddBinaries = @()
foreach ($ext in $Extensions) {
    $AddBinaries += @("--add-binary", "$(Join-Path $Stage "$ext$ExtSuffix");.")
}
$OpenMpDll = Join-Path $Stage "vcomp140.dll"
if (Test-Path $OpenMpDll) { $AddBinaries += @("--add-binary", "$OpenMpDll;.") }

Write-Host "==> Running PyInstaller"
$Bundle = Join-Path $Dist "ChemInsight3D"
if (Test-Path $Bundle) { Remove-Item -Recurse -Force $Bundle }
Invoke-Checked $VenvPython (@(
    "-m", "PyInstaller",
    "--noconfirm", "--onedir", "--windowed",
    "--name", "ChemInsight3D",
    "--distpath", $Dist,
    "--workpath", (Join-Path $WorkDir "pyinstaller"),
    "--specpath", $WorkDir,
    "--paths", $Stage
) + $AddBinaries + @(
    "--hidden-import", "electron_density_opt_omp",
    "--hidden-import", "localization_native",
    "--exclude-module", "tkinter",
    "--exclude-module", "pytest",
    (Join-Path $Stage "ChemInsight3D.py")
))

Write-Host "==> Checking the C++ extensions load inside the bundle"
# A windowed app prints nothing, so the result is read from its exit code.
$Exe = Join-Path $Bundle "ChemInsight3D.exe"
$Check = Start-Process -FilePath $Exe -ArgumentList "--self-check" -Wait -PassThru
if ($Check.ExitCode -ne 0) { throw "Self-check failed: a C++ extension did not load (exit $($Check.ExitCode))." }
Write-Host "    All C++ extensions load."

if ($Zip) {
    Write-Host "==> Creating zip"
    $ZipPath = Join-Path $WorkDir "ChemInsight3D-windows.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Compress-Archive -Path $Bundle -DestinationPath $ZipPath
    Write-Host "    $ZipPath"
}

$SizeGB = (Get-ChildItem -Recurse $Bundle | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ""
Write-Host ("==> Done: {0} ({1:N1} GB)" -f $Exe, $SizeGB)
