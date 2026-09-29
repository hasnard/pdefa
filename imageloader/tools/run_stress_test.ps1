# 100 interlaced PNG'yi decoder'dan gecir, orijinalleri ile karsilastir
$ErrorActionPreference = "Continue"
$dir = "D:\pngtest"
$to_ppm = "D:\PROJE\engineforaiincpu\imageloader\build\Debug\png_to_ppm.exe"
$tmp_o = "$env:TEMP\test_orig.ppm"
$tmp_i = "$env:TEMP\test_int.ppm"

$pass = 0
$fail = 0
$err  = 0

Get-ChildItem "$dir\*_orig.png" | Sort-Object Name | ForEach-Object {
    $orig = $_.FullName
    $name = $_.BaseName -replace "_orig$", ""
    $int  = "$dir\${name}_int.png"

    # Decoder her iki PNG'yi PPM'e cevirsin
    & $to_ppm $orig $tmp_o 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERR ] $name (orig decode)" -ForegroundColor DarkRed
        $err++
        return
    }

    & $to_ppm $int $tmp_i 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERR ] $name (interlaced decode)" -ForegroundColor DarkRed
        $err++
        return
    }

    # PPM'leri byte-byte karsilastir
    $a = [IO.File]::ReadAllBytes($tmp_o)
    $b = [IO.File]::ReadAllBytes($tmp_i)

    $same = $true
    if ($a.Length -ne $b.Length) {
        $same = $false
    } else {
        for ($i = 0; $i -lt $a.Length; $i++) {
            if ($a[$i] -ne $b[$i]) { $same = $false; break }
        }
    }

    if ($same) {
        Write-Host "[ OK ] $name" -ForegroundColor Green
        $pass++
    } else {
        Write-Host "[FAIL] $name" -ForegroundColor Red
        $fail++
    }
}

Write-Host ""
Write-Host "====== SONUC ======" -ForegroundColor Cyan
Write-Host "Gecen   : $pass" -ForegroundColor Green
Write-Host "Basarisiz: $fail" -ForegroundColor Red
Write-Host "Hata    : $err"  -ForegroundColor Yellow
Write-Host "Toplam  : $($pass + $fail + $err)"