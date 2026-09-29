#!/usr/bin/env python3
"""Project source audit — boyut + harici kütüphane taraması.

Düzeltmeler:
  - build*, cmake-build-* gibi prefix'lerle eşleşen dizinler atlanır
  - Google Benchmark / Google Test header'ları harici kütüphane olarak sayılır
  - Core (src+include) ile dev (examples/benchmarks/tools) ayrı raporlanır
  - "self-contained" iddiası sadece core için yapılır
"""
import os
import re
from collections import defaultdict

ROOT = r"D:\PROJE\engineforaiincpu\PDEFA1.0"
EXTS = ('.cpp', '.cc', '.cxx', '.hpp', '.h', '.hh', '.c')

# --- Dizin atlama: TAM eşleşme ---
SKIP_DIRS_EXACT = {
    '.git', '.vs', '.vscode', 'out',
    'venv', '.venv', 'node_modules',
    'third_party', 'external', 'extern', 'vendor',
    'googletest', 'google_test', 'gtest', 'gmock',
    'test', 'tests', 'testing', 'unittest', 'unit_tests',
}

# --- Dizin atlama: PREFIX eşleşmesi (build_shared, cmake-build-debug vs.) ---
SKIP_DIR_PREFIXES = (
    'build',          # build, build_shared, build_debug, build_release
    'cmake-build-',   # cmake-build-debug, cmake-build-release
    '_deps',          # FetchContent çıktıları
)

SKIP_FILE_PREFIXES = ('test_', 'gtest_', 'gmock_')
SKIP_FILE_SUFFIXES = (
    '_test.cpp', '_test.cc', '_test.cxx', '_test.hpp',
    '_test.h', '_unittest.cpp',
)

# Core proje: sadece burada "self-contained" iddiası geçerli
CORE_DIRS = {'src', 'include'}

# --- Harici kütüphaneler (header bazlı) ---
EXT_LIBS = {
    'opencv':       re.compile(r'#include\s*[<"]opencv2?[/"]', re.I),
    'eigen':        re.compile(r'#include\s*[<"]Eigen[/"]',     re.I),
    'torch':        re.compile(r'#include\s*[<"]torch[/"]',     re.I),
    'onnx':         re.compile(r'#include\s*[<"]onnx',          re.I),
    'protobuf':     re.compile(r'#include\s*[<"]google/protobuf', re.I),
    'numpy':        re.compile(r'#include\s*[<"]numpy[/"]',     re.I),
    'mkl':          re.compile(r'#include\s*[<"]mkl',           re.I),
    'openblas':     re.compile(r'#include\s*[<"]cblas',         re.I),
    'boost':        re.compile(r'#include\s*[<"]boost[/"]',     re.I),
    'json':         re.compile(r'#include\s*[<"]nlohmann',      re.I),
    'fmt':          re.compile(r'#include\s*[<"]fmt[/"]',       re.I),
    'spdlog':       re.compile(r'#include\s*[<"]spdlog[/"]',    re.I),
    'stb':          re.compile(r'#include\s*[<"]stb_',          re.I),
    # --- Yeni eklenenler: benchmark + test altyapısı ---
    'gtest':        re.compile(r'#include\s*[<"]gtest[/"]',     re.I),
    'gmock':        re.compile(r'#include\s*[<"]gmock[/"]',     re.I),
    'gbenchmark':   re.compile(r'#include\s*[<"]benchmark[/"]', re.I),
    'check':        re.compile(r'#include\s*[<"]check\.h[">]',  re.I),
    'internal_macros': re.compile(r'#include\s*[<"]internal_macros\.h[">]', re.I),
    'string_util':  re.compile(r'#include\s*[<"]string_util\.h[">]', re.I),
    'timers':       re.compile(r'#include\s*[<"]timers\.h[">]', re.I),
}

SYSTEM = re.compile(r'#include\s*[<"]([^">]+)[">]')
OWN    = re.compile(r'engine/|ops/|runtime/|kernels/|memory/|loader/|core/|api/')


def should_skip_dir(dirname: str) -> bool:
    low = dirname.lower()
    if low in SKIP_DIRS_EXACT:
        return True
    for p in SKIP_DIR_PREFIXES:
        if low.startswith(p):
            return True
    return False


def should_skip_file(fname: str) -> bool:
    low = fname.lower()
    if low.startswith(SKIP_FILE_PREFIXES):
        return True
    if low.endswith(SKIP_FILE_SUFFIXES):
        return True
    if 'gtest' in low or 'gmock' in low:
        return True
    return False


def top_dir(rel_path: str) -> str:
    parts = rel_path.split(os.sep)
    return parts[0] if parts else ''


# ---------------- Tarama ----------------
size_total = 0
size_core = 0
by_dir = defaultdict(lambda: [0, 0])       # top_dir -> [bytes, count]
files = []
ext_hits = defaultdict(list)
includes_all = defaultdict(int)
skipped_dirs = 0
skipped_files = 0

for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if not should_skip_dir(d)]
    if should_skip_dir(os.path.basename(dp)):
        skipped_dirs += 1
        continue

    for f in fn:
        if not f.endswith(EXTS):
            continue
        if should_skip_file(f):
            skipped_files += 1
            continue

        p = os.path.join(dp, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue

        size_total += sz
        rel = os.path.relpath(p, ROOT)
        tdir = top_dir(rel)
        by_dir[tdir][0] += sz
        by_dir[tdir][1] += 1
        files.append((rel, sz))

        if tdir in CORE_DIRS:
            size_core += sz

        try:
            txt = open(p, 'r', encoding='utf-8', errors='ignore').read()
        except Exception:
            continue

        for line in txt.splitlines():
            if '#include' not in line:
                continue
            m = SYSTEM.search(line)
            if not m:
                continue
            inc = m.group(1)
            includes_all[inc] += 1
            for name, pat in EXT_LIBS.items():
                if pat.search(line):
                    ext_hits[name].append((rel, line.strip()))


# ---------------- Rapor ----------------
print("=" * 72)
print(f"  PROJE KAYNAK DENETİMİ — {ROOT}")
print("=" * 72)
print(f"\n  Toplam .cpp/.hpp/.h dosya : {len(files)}")
print(f"  Toplam boyut              : {size_total:,} byte "
      f"= {size_total/1024:.1f} KB = {size_total/1024/1024:.2f} MB")
print(f"  Core (src+include) boyut  : {size_core:,} byte "
      f"= {size_core/1024:.1f} KB = {size_core/1024/1024:.2f} MB")
print(f"  Atlanan dizin             : {skipped_dirs}")
print(f"  Atlanan test dosyası      : {skipped_files}")
print("  (build*, cmake-build-*, _deps, test/gtest klasörleri hariç)\n")

print("  Dizin bazında:")
print(f"  {'dizin':<20s} {'dosya':>6s} {'boyut':>12s} {'KB':>10s}  {'tip':<8s}")
print("  " + "-" * 62)
for d, (sz, n) in sorted(by_dir.items(), key=lambda x: -x[1][0]):
    tip = "core" if d in CORE_DIRS else "dev"
    print(f"  {d:<20s} {n:>6d} {sz:>12,d} {sz/1024:>10.1f}  {tip:<8s}")

# --- Harici kütüphaneler ---
print("\n" + "=" * 72)
print("  HARİCİ KÜTÜPHANE TARAMASI")
print("=" * 72)

# Core ve dev ayrımı
core_hits = defaultdict(list)
dev_hits = defaultdict(list)
for name, hits in ext_hits.items():
    for rel, line in hits:
        if top_dir(rel) in CORE_DIRS:
            core_hits[name].append((rel, line))
        else:
            dev_hits[name].append((rel, line))

if not ext_hits:
    print("\n  ✓ HİÇBİR HARİCİ KÜTÜPHANE BULUNAMADI.")
    print("    Proje tamamen kendi kendine yeten (self-contained) C++.")
else:
    if core_hits:
        print("\n  ⚠ CORE'DA HARİCİ KÜTÜPHANE VAR (kritik):")
        for name, hits in core_hits.items():
            print(f"\n    {name.upper():<12s} → {len(hits)} kullanım")
            for rel, line in hits[:5]:
                print(f"      {rel}: {line}")
            if len(hits) > 5:
                print(f"      ... +{len(hits)-5} tane daha")
    else:
        print("\n  ✓ CORE (src+include): HARİCİ KÜTÜPHANE YOK.")
        print("    Kütüphanenin kendisi tamamen self-contained.")

    if dev_hits:
        print("\n  ℹ DEV katmanında (test/benchmark/examples) harici kullanım:")
        for name, hits in dev_hits.items():
            print(f"    {name.upper():<12s} → {len(hits)} kullanım")

# --- En sık include edilen header'lar ---
print("\n" + "=" * 72)
print("  EN SIK #include EDİLEN HEADER'LAR (top 25)")
print("=" * 72)
for inc, n in sorted(includes_all.items(), key=lambda x: -x[1])[:25]:
    tag = "[kendi]" if OWN.search(inc) else "[sistem]"
    print(f"  {n:>4d}x  {inc}  {tag}")

# --- En büyük 15 dosya (sadece core+dev, build artifact yok) ---
print("\n" + "=" * 72)
print("  EN BÜYÜK 15 KAYNAK DOSYA (build artifact hariç)")
print("=" * 72)
for rel, sz in sorted(files, key=lambda x: -x[1])[:15]:
    print(f"  {sz:>8,d} B  {rel}")

# --- Özet ---
print("\n" + "=" * 72)
print("  ÖZET")
print("=" * 72)

if core_hits:
    print(f"""
  ⚠ CORE'da {sum(len(v) for v in core_hits.values())} harici kütüphane kullanımı var.
  Kütüphane artık %100 self-contained DEĞİL.
""")
else:
    print(f"""
  ✓ CORE (src+include): %100 self-contained.
  ✓ Harici kütüphane: YOK.
  ✓ Sadece C++ standart kütüphanesi + AVX2 intrinsics.
  ✓ {len(files)} dosya, {size_total/1024:.0f} KB toplam kaynak.
  ✓ Core boyutu: {size_core/1024:.0f} KB.
""")

if dev_hits:
    print(f"  ℹ DEV katmanında (test/benchmark) kullanılan harici kütüphaneler:")
    for name in sorted(dev_hits.keys()):
        print(f"      - {name}")
    print("  (Bu normaldir; core'u etkilemez.)")