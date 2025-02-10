import os
import logging

def inisialisasi_folder():
    """
    Membuat struktur folder yang diperlukan untuk preprocessing dan vektorisasi
    """
    try:
        # Daftar folder yang diperlukan
        folders = [
            'data/processed',
            'data/vectorized',
            'models/word2vec'
        ]
        
        # Buat folder jika belum ada
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Folder {folder} berhasil dibuat")
            else:
                print(f"Folder {folder} sudah ada")
                
        return True
        
    except Exception as e:
        print(f"Terjadi kesalahan saat membuat folder: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Jalankan inisialisasi
    if inisialisasi_folder():
        print("Inisialisasi folder berhasil")
    else:
        print("Inisialisasi folder gagal")