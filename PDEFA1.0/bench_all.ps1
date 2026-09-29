# ============================================================
#  bench_all.ps1 - Full test suite
#  Engine-AI vs PyTorch vs ONNX Runtime
#  Usage: .\bench_all.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$ROOT = "D:\PROJE\engineforaiincpu\PDEFA1.0"

function Print-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
}

function Print-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "--- $Title ---" -ForegroundColor Yellow
}

function Print-OK   { param([string]$Msg) Write-Host "  [OK]   $Msg" -ForegroundColor Green }
function Print-FAIL { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red }
function Print-INFO { param([string]$Msg) Write-Host "  [INFO] $Msg" -ForegroundColor Gray }

# ============================================================
#  START
# ============================================================
Clear-Host
Print-Header "ENGINE-AI FULL TEST SUITE"

# ============================================================
#  0. SYSTEM INFO
# ============================================================
Print-Section "0. System Info"
$cpu = Get-CimInstance -ClassName Win32_Processor
Print-INFO "CPU: $($cpu.Name)"
Print-INFO "Cores: $($cpu.NumberOfCores) physical / $($cpu.NumberOfLogicalProcessors) logical"

$battery = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue
if ($battery) {
    Print-INFO "Battery: $($battery.EstimatedChargeRemaining) percent"
    if ($battery.BatteryStatus -eq 1) {
        Write-Host "  [WARN] BATTERY MODE - performance will be lower!" -ForegroundColor Red
        Write-Host "         TIP: Plug in charger, wait 5 min, re-run." -ForegroundColor Yellow
    }
}

$power = (powercfg /getactivescheme)
Print-INFO "Power: $power"

# ============================================================
#  1. BUILD
# ============================================================
Print-Section "1. Building"
Set-Location $ROOT
cmake --build build --config Release -j 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Print-FAIL "build error"
    exit 1
}
Print-OK "build complete"

# ============================================================
#  2. COPY MODELS
# ============================================================
Print-Section "2. Copying models"
Copy-Item tools\converter\mini_detector.engine  build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\simple_cnn.engine     build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\simple_cnn_input.f32  build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\simple_cnn_output.f32 build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\bn_cnn.engine         build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\bn_cnn_input.f32      build\bin\Release\ -Force -ErrorAction SilentlyContinue
Copy-Item tools\converter\bn_cnn_output.f32     build\bin\Release\ -Force -ErrorAction SilentlyContinue
Print-OK "models copied"

# ============================================================
#  3. CORRECTNESS TESTS
# ============================================================
Print-Header "PART 1: CORRECTNESS TESTS"
Set-Location "$ROOT\build\bin\Release"

# --- Smoke test ---
Print-Section "3.1 Smoke Test"
.\smoke_test.exe

# --- Conv2d correctness ---
Print-Section "3.2 Conv2d Correctness"
.\test_conv2d.exe

# --- Simple CNN (with PyTorch comparison) ---
Print-Section "3.3 Simple CNN (PyTorch comparison)"
.\test_engine_load.exe

# --- BatchNorm CNN ---
Print-Section "3.4 BatchNorm CNN (PyTorch comparison)"
.\test_bn_load.exe

# --- Memory tests ---
Print-Section "3.5 Memory Tests"
if (Test-Path ".\test_memory.exe") {
    .\test_memory.exe
} else {
    Print-INFO "test_memory.exe not found - skipping"
}

# ============================================================
#  4. MATMUL BENCHMARKS
# ============================================================
Print-Header "PART 2: MATMUL BENCHMARKS"

Print-Section "4.1 Engine-AI Matmul"
.\benchmark_matmul.exe

Print-Section "4.2 PyTorch Matmul"
Set-Location "$ROOT\tools\benchmark"
if (Test-Path "compare_pytorch.py") {
    python compare_pytorch.py
} else {
    Print-INFO "compare_pytorch.py not found"
}

Print-Section "4.3 ONNX Runtime Matmul"
if (Test-Path "compare_onnx.py") {
    python compare_onnx.py
} else {
    Print-INFO "compare_onnx.py not found"
}

# ============================================================
#  5. CONV2D BENCHMARKS
# ============================================================
Print-Header "PART 3: CONV2D BENCHMARKS"
Set-Location "$ROOT\build\bin\Release"

Print-Section "5.1 Engine-AI Conv2d (YOLOv5n layers)"
.\benchmark_conv2d.exe

# ============================================================
#  6. DETECTION MODEL BENCHMARKS
# ============================================================
Print-Header "PART 4: DETECTION MODEL (MiniDetector 640x640)"

Print-Section "6.1 Engine-AI Detection"
.\benchmark_detection.exe

Print-Section "6.2 PyTorch + ONNX Runtime Detection"
Set-Location "$ROOT\tools\benchmark"
if (Test-Path "compare_detection.py") {
    python compare_detection.py
} else {
    Print-INFO "compare_detection.py not found"
}

# ============================================================
#  SUMMARY
# ============================================================
Print-Header "ALL TESTS COMPLETE"
Set-Location $ROOT
Print-INFO "Check output above for results"
Write-Host ""