import pandas as pd
import numpy as np
import logging
import torch
from preprocessing.text_preprocessor import TextPreprocessor
from utils.vector_storage_manager import VectorStorageManager

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def konversi_vektor(vektor):
    """
    Konversi vektor ke tipe numpy yang konsisten
    """
    if isinstance(vektor, torch.Tensor):
        return vektor.numpy()
    elif isinstance(vektor, np.ndarray):
        return vektor
    else:
        try:
            return np.array(vektor)
        except Exception as e:
            logger.error(f"Gagal mengkonversi vektor: {e}")
            return None

def validasi_dan_vektorisasi_data(
    path_csv='data/Dataset Twitter.csv', 
    konfigurasi_preprocessing=None, 
    metode_vektorisasi='word2vec'):
    """
    Preprocessing dan vektorisasi data dengan validasi lanjutan
    """
    # Konfigurasi preprocessing default
    if konfigurasi_preprocessing is None:
        konfigurasi_preprocessing = {
            'remove_mentions': True,
            'remove_urls': True,
            'remove_hashtags': True,
            'remove_numbers': True,
            'remove_special_chars': True,
            'remove_emojis': True,
            'case_folding': True,
            'remove_stopwords': True,
            'stemming': True
        }
    
    # Muat data
    try:
        df = pd.read_csv(path_csv)
        teks = df['tweet'].tolist()
        logger.info(f"Memuat dataset: {len(teks)} teks")
    except Exception as e:
        logger.error(f"Kesalahan memuat dataset: {e}")
        return None
    
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor()
    
    try:
        # Proses dan vektorisasi teks
        logger.info("Memulai proses preprocessing dan vektorisasi...")
        
        # Preprocessing teks
        teks_diproses = preprocessor._process_texts_with_stats(
            teks, 
            konfigurasi_preprocessing
        )
        logger.info(f"Preprocessing selesai. Jumlah teks: {len(teks_diproses)}")
        
        # Vektorisasi
        try:
            vektor = preprocessor.vectorize_text(
                teks_diproses, 
                method=metode_vektorisasi
            )
            
            # Konversi vektor ke tipe yang konsisten
            vektor_final = konversi_vektor(vektor)
            
            if vektor_final is None:
                raise ValueError("Gagal mengkonversi vektor")
            
            logger.info(f"Vektorisasi berhasil. Bentuk vektor: {vektor_final.shape}")
        except Exception as e:
            logger.error(f"Kesalahan vektorisasi: {e}")
            return None
        
        # Pemeriksaan validasi tambahan
        if vektor_final is None:
            logger.error("Vektorisasi gagal: Tidak ada vektor yang dihasilkan")
            return None
        
        # Periksa dimensi dan elemen non-nol
        if vektor_final.size == 0:
            logger.error("Kesalahan: Vektor yang dihasilkan kosong")
            return None
        
        # Periksa dan tangani nilai NaN atau tak terhingga
        if np.isnan(vektor_final).any() or np.isinf(vektor_final).any():
            logger.warning("Peringatan: Vektor mengandung nilai NaN atau tak terhingga")
            vektor_final = np.nan_to_num(vektor_final)  # Ganti NaN/inf dengan nol
        
        # Laporan vektorisasi dengan penanganan kesalahan
        try:
            # Modifikasi get_vectorization_report untuk menangani berbagai tipe vektor
            def laporan_manual(teks):
                return {
                    'Metode': metode_vektorisasi,
                    'Dimensi Vektor': vektor_final.shape[1],
                    'Jumlah Sampel': vektor_final.shape[0],
                    'Tipe Data': str(vektor_final.dtype)
                }
            
            laporan_vektorisasi = laporan_manual(teks_diproses)
            logger.info("Laporan Vektorisasi:")
            for kunci, nilai in laporan_vektorisasi.items():
                logger.info(f"{kunci}: {nilai}")
        except Exception as e:
            logger.warning(f"Gagal membuat laporan vektorisasi: {e}")
        
        return {
            'vektor': vektor_final,
            'teks_diproses': teks_diproses,
            'konfigurasi_preprocessing': konfigurasi_preprocessing,
            'metode_vektorisasi': metode_vektorisasi
        }
    
    except Exception as e:
        logger.error(f"Kesalahan vektorisasi komprehensif: {e}")
        return None

def main():
    # Tambahkan opsi untuk memilih metode vektorisasi
    metode_vektorisasi = ['tfidf', 'word2vec', 'bert']
    
    for metode in metode_vektorisasi:
        print(f"\n--- Mencoba Vektorisasi dengan Metode: {metode} ---")
        hasil = validasi_dan_vektorisasi_data(metode_vektorisasi=metode)
        
        if hasil:
            print(f"Vektorisasi berhasil dengan metode {metode}!")
            print(f"Bentuk vektor: {hasil['vektor'].shape}")
            print(f"Jumlah teks diproses: {len(hasil['teks_diproses'])}")
        else:
            print(f"Vektorisasi dengan metode {metode} mengalami masalah.")

if __name__ == '__main__':
    main()