from pathlib import Path

def scan_and_save_structure():
    # Çalışılan dizinde "engine-ai" var mı yoksa doğrudan bulunduğumuz yer mi?
    base_dir = Path("engine-ai")
    
    if not base_dir.exists():
        base_dir = Path(".")
    
    target_dirs = [base_dir / "include", base_dir / "src"]
    output_txt = base_dir / "include_src_structure.txt"
    
    # Çıktı dosyasının klasörünün var olduğundan emin ol
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    
    content = []
    content.append("=== DİNAMİK DOSYA YAPISI VE BOYUT RAPORU ===\n")
    content.append(f"Tarama Konumu: {base_dir.resolve()}\n")
    
    found_target = False
    for target in target_dirs:
        if not target.exists():
            content.append(f"\n[ATLANDI] {target.name} dizini bulunamadı.")
            continue
            
        found_target = True
        content.append(f"\n[{target.name.upper()} DİZİNİ]")
        
        # Klasör altındaki tüm dosya ve klasörleri tara
        for path in sorted(target.rglob("*")):
            try:
                relative_path = path.relative_to(base_dir)
            except ValueError:
                relative_path = path
                
            if path.is_dir():
                content.append(f"    📁 {relative_path}/")
            else:
                # Dosya boyutunu byte cinsinden al
                file_size = path.stat().st_size
                if file_size == 0:
                    content.append(f"    📄 {relative_path}  ➡️  [0 KB / Boş]")
                else:
                    # Dolu dosyalar için isteğe bağlı boyut gösterimi (KB cinsinden)
                    size_kb = file_size / 1024
                    if size_kb < 1:
                        content.append(f"    📄 {relative_path}  ({file_size} bytes)")
                    else:
                        content.append(f"    📄 {relative_path}  ({size_kb:.2f} KB)")

    if not found_target:
        content.append("\n[UYARI] Ne 'include' ne de 'src' dizini bulunabildi!")

    # Güvenli bir şekilde dosyaya yaz
    output_txt.write_text("\n".join(content), encoding="utf-8")
    print(f"✅ Tarama tamamlandı! Boyut raporu oluşturuldu: {output_txt.resolve()}")

if __name__ == "__main__":
    scan_and_save_structure()