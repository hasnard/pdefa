# test_exe.py
import subprocess, os
EXE = r"D:\PROJE\engineforaiincpu\PDEFA1.0\build\bin\Release\benchmark_matmul.exe"
print("exe var mi?:", os.path.exists(EXE))
out = subprocess.check_output([EXE], text=True, cwd=os.path.dirname(EXE),
                              stderr=subprocess.STDOUT, timeout=300)
print("=== RAW OUTPUT (repr) ===")
print(repr(out[:600]))
print("=== NORMAL OUTPUT ===")
print(out)