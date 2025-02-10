import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import glob
import os
from utils.storage_manager import StorageManager

def compare_preprocessing_methods(preprocessor, texts: List[str], labels: List[str]):
    """
    Membandingkan hasil preprocessing dengan berbagai metode vektorisasi
    """
    results = []
    methods = ['tfidf', 'word2vec', 'bert']
    
    # Konfigurasi preprocessing default
    preprocessing_config = {
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
    
    # Progress bar
    progress_text = "Membandingkan metode preprocessing..."
    progress_bar = st.progress(0)
    
    for idx, method in enumerate(methods):
        try:
            st.info(f"Processing dengan {method.upper()}...")
            
            # Preprocessing dan vektorisasi
            vectors, version_id = preprocessor.process_and_vectorize(
                texts,
                preprocessing_config,
                method
            )
            
            # Analisis hasil
            if vectors is not None:
                vocab_size = preprocessor.get_vectorization_report([texts])[method]['Vector Dimension']
                sparsity = preprocessor.get_vectorization_report([texts])[method]['Sparsity']
                
                results.append({
                    'Method': method.upper(),
                    'Vector Dimension': vocab_size,
                    'Sparsity': sparsity,
                    'Processing Time': preprocessor.get_vectorization_report([texts])[method].get('Processing Time', 0)
                })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(methods))
            
        except Exception as e:
            st.error(f"Error processing {method}: {str(e)}")
            
    progress_bar.empty()
    
    return pd.DataFrame(results)

def visualize_comparison(comparison_df: pd.DataFrame):
    """
    Visualisasi perbandingan preprocessing
    """
    st.subheader("Perbandingan Metode Preprocessing")
    
    # 1. Dimensi Vektor
    fig_dim = px.bar(
        comparison_df,
        x='Method',
        y='Vector Dimension',
        title='Perbandingan Dimensi Vektor',
        color='Method'
    )
    st.plotly_chart(fig_dim)
    
    # 2. Sparsity
    fig_sparsity = px.bar(
        comparison_df,
        x='Method',
        y='Sparsity',
        title='Perbandingan Sparsity (Tingkat Kejarangan)',
        color='Method'
    )
    st.plotly_chart(fig_sparsity)
    
    # 3. Processing Time
    fig_time = px.bar(
        comparison_df,
        x='Method',
        y='Processing Time',
        title='Perbandingan Waktu Pemrosesan',
        color='Method'
    )
    st.plotly_chart(fig_time)
    
    # 4. Radar Chart
    fig_radar = go.Figure()
    metrics = ['Vector Dimension', 'Sparsity', 'Processing Time']
    normalized_data = comparison_df.copy()
    
    for metric in metrics:
        max_val = normalized_data[metric].max()
        if max_val != 0:
            normalized_data[metric] = normalized_data[metric] / max_val
            
    for method in comparison_df['Method']:
        method_data = normalized_data[normalized_data['Method'] == method]
        fig_radar.add_trace(go.Scatterpolar(
            r=method_data[metrics].values[0],
            theta=metrics,
            name=method,
            fill='toself'
        ))
        
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Perbandingan Karakteristik Metode Preprocessing'
    )
    st.plotly_chart(fig_radar)
    
    # Tampilkan tabel perbandingan
    st.subheader("Tabel Perbandingan Detail")
    st.dataframe(comparison_df)
    
    # Analisis dan rekomendasi
    st.subheader("Analisis dan Rekomendasi")
    
    best_dimension = comparison_df.loc[comparison_df['Vector Dimension'].idxmax(), 'Method']
    best_sparsity = comparison_df.loc[comparison_df['Sparsity'].idxmin(), 'Method']
    best_time = comparison_df.loc[comparison_df['Processing Time'].idxmin(), 'Method']
    
    st.write(f"""
    **Analisis:**
    - Dimensi Vektor Tertinggi: {best_dimension}
    - Sparsity Terendah: {best_sparsity}
    - Waktu Pemrosesan Tercepat: {best_time}
    
    **Rekomendasi:**
    - Untuk dataset besar: {'TF-IDF' if best_time == 'TFIDF' else best_time}
    - Untuk akurasi tinggi: {'BERT' if 'BERT' in comparison_df['Method'].values else best_dimension}
    - Untuk keseimbangan: {'WORD2VEC' if 'WORD2VEC' in comparison_df['Method'].values else best_sparsity}
    """)

def load_and_compare_preprocessed():
    """Load dan bandingkan hasil preprocessing yang sudah ada"""
    storage = StorageManager()
    
    # Ambil semua file preprocessed yang tersedia
    processed_files = glob.glob('data/processed/preprocessed_*.npz')
    if not processed_files:
        st.warning("Tidak ada data preprocessing yang tersedia.")
        return
        
    # Ekstrak version ID lengkap dari nama file
    versions = [
        os.path.basename(f).replace('preprocessed_', '').replace('.npz', '')
        for f in processed_files
    ]
    
    selected_versions = st.multiselect(
        "Pilih 3 versi preprocessing untuk dibandingkan:",
        versions,
        max_selections=3
    )
    
    if len(selected_versions) > 0:
        comparison_results = []
        
        for version in selected_versions:
            try:
                # Gunakan load_preprocessed_with_vectors yang baru
                texts, vectors, metadata = storage.load_preprocessed_with_vectors(version)
                
                if texts is not None and vectors is not None and metadata is not None:
                    # Hitung metrik
                    result = {
                        'Version': version,
                        'Method': metadata.get('method', 'Unknown').upper(),
                        'Vector Dimension': vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0],
                        'Vocab Size': len(set(' '.join(texts).split())),
                        'Avg Text Length': np.mean([len(text.split()) for text in texts])
                    }
                    
                    # Tambahkan statistik sparsity
                    if metadata.get('is_sparse', False):
                        sparsity = 1.0 - (np.count_nonzero(vectors) / vectors.size)
                    else:
                        sparsity = 1.0 - (np.count_nonzero(vectors) / vectors.size)
                    
                    result['Sparsity'] = sparsity
                    comparison_results.append(result)
                else:
                    st.error(f"Error loading data for version {version}")
                    
            except Exception as e:
                st.error(f"Error loading version {version}: {str(e)}")
                
        if comparison_results:
            df = pd.DataFrame(comparison_results)
            visualize_preprocessing_comparison(df)

# Fungsi untuk visualisasi perbandingan preprocessing yang sudah ada
def visualize_preprocessing_comparison(df: pd.DataFrame):
    """
    Visualisasi perbandingan preprocessing
    """
    # 1. Bar chart untuk dimensi vektor
    fig_dim = px.bar(
        df,
        x='Version',
        y='Vector Dimension',
        color='Method',
        title='Perbandingan Dimensi Vektor',
        labels={'Vector Dimension': 'Dimensi', 'Version': 'Versi'}
    )
    st.plotly_chart(fig_dim)
    
    # 2. Bar chart untuk vocabulary size
    fig_vocab = px.bar(
        df,
        x='Version',
        y='Vocab Size',
        color='Method',
        title='Perbandingan Ukuran Vocabulary',
        labels={'Vocab Size': 'Jumlah Kata Unik', 'Version': 'Versi'}
    )
    st.plotly_chart(fig_vocab)
    
    # 3. Radar chart untuk perbandingan keseluruhan
    metrics = ['Vector Dimension', 'Vocab Size', 'Avg Text Length']
    fig_radar = go.Figure()
    
    # Normalisasi data untuk radar chart
    normalized_df = df.copy()
    for metric in metrics:
        max_val = normalized_df[metric].max()
        if max_val != 0:
            normalized_df[metric] = normalized_df[metric] / max_val
    
    for _, row in normalized_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            name=f"{row['Method']} ({row['Version']})",
            fill='toself'
        ))
        
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Perbandingan Karakteristik Preprocessing'
    )
    st.plotly_chart(fig_radar)
    
    # 4. Tabel perbandingan detail
    st.subheader("Detail Perbandingan")
    st.dataframe(df.style.highlight_max(axis=0, props='background-color: #90EE90'))
    
    # 5. Analisis perbandingan
    st.subheader("Analisis Perbandingan")
    
    best_dimension = df.loc[df['Vector Dimension'].idxmax()]
    best_vocab = df.loc[df['Vocab Size'].idxmax()]
    most_sparse = df.loc[df['Sparsity'].idxmax()]
    
    st.write(f"""
    **Analisis Dimensi Vector:**
    - Dimensi tertinggi: {best_dimension['Method']} (Versi: {best_dimension['Version']})
    - Dimensi: {best_dimension['Vector Dimension']}
    
    **Analisis Vocabulary:**
    - Vocabulary terbesar: {best_vocab['Method']} (Versi: {best_vocab['Version']})
    - Jumlah kata unik: {best_vocab['Vocab Size']}
    
    **Analisis Sparsity:**
    - Metode dengan sparsity tertinggi: {most_sparse['Method']} (Versi: {most_sparse['Version']})
    - Sparsity: {most_sparse['Sparsity']:.2%}
    """)
    
    # Tambahkan bagian evaluasi metode
    st.subheader("Evaluasi Detail Setiap Metode")
    
    # Evaluasi TF-IDF
    st.write("### 1. TF-IDF")
    tfidf_data = df[df['Method'] == 'TFIDF'].iloc[0] if 'TFIDF' in df['Method'].values else None
    if tfidf_data is not None:
        st.write("""
        **Karakteristik:**
        - Dimensi Vektor: {} ({}%)
        - Vocabulary Size: {}
        - Sparsity: {:.2%}
        
        **Kelebihan:**
        - Mampu menangkap banyak fitur dan vocabulary
        - Baik untuk menangkap kata-kata penting dalam dokumen
        
        **Kekurangan:**
        - Sparsity tinggi yang dapat menyebabkan komputasi berat
        - Membutuhkan memori lebih besar
        """.format(
            tfidf_data['Vector Dimension'],
            100,
            tfidf_data['Vocab Size'],
            tfidf_data['Sparsity']
        ))

    # Evaluasi Word2Vec
    st.write("### 2. Word2Vec")
    w2v_data = df[df['Method'] == 'WORD2VEC'].iloc[0] if 'WORD2VEC' in df['Method'].values else None
    if w2v_data is not None:
        st.write("""
        **Karakteristik:**
        - Dimensi Vektor: {} ({}%)
        - Vocabulary Size: {}
        - Sparsity: {:.2%}
        
        **Kelebihan:**
        - Dense vectors (tidak sparse)
        - Efisien untuk komputasi
        - Baik dalam menangkap hubungan semantik
        
        **Kekurangan:**
        - Dimensi lebih kecil
        - Mungkin kehilangan beberapa informasi detail
        """.format(
            w2v_data['Vector Dimension'],
            (w2v_data['Vector Dimension'] / df['Vector Dimension'].max()) * 100,
            w2v_data['Vocab Size'],
            w2v_data['Sparsity']
        ))

    # Evaluasi BERT
    st.write("### 3. BERT")
    bert_data = df[df['Method'] == 'BERT'].iloc[0] if 'BERT' in df['Method'].values else None
    if bert_data is not None:
        st.write("""
        **Karakteristik:**
        - Dimensi Vektor: {} ({}%)
        - Vocabulary Size: {}
        - Sparsity: {:.2%}
        
        **Kelebihan:**
        - Representasi kontekstual yang kaya
        - Mampu menangkap konteks kalimat dengan baik
        
        **Kekurangan:**
        - Dimensi paling kecil
        - Komputasi lebih berat saat preprocessing
        """.format(
            bert_data['Vector Dimension'],
            (bert_data['Vector Dimension'] / df['Vector Dimension'].max()) * 100,
            bert_data['Vocab Size'],
            bert_data['Sparsity']
        ))

    # Rekomendasi berdasarkan analisis
    st.write("### 📊 Rekomendasi")
    
    # Hitung skor sederhana untuk setiap metode
    method_scores = {}
    max_dimension = df['Vector Dimension'].max()
    
    for _, row in df.iterrows():
        method = row['Method']
        # Skor berdasarkan:
        # 1. Rasio dimensi (30%)
        # 2. Inverse sparsity (40%)
        # 3. Efisiensi komputasi berdasarkan dimensi (30%)
        dimension_score = (row['Vector Dimension'] / max_dimension) * 0.3
        sparsity_score = (1 - row['Sparsity']) * 0.4
        efficiency_score = (1 - (row['Vector Dimension'] / max_dimension)) * 0.3
        
        total_score = dimension_score + sparsity_score + efficiency_score
        method_scores[method] = total_score
    
    best_method = max(method_scores.items(), key=lambda x: x[1])[0]
    
    st.write(f"""
    **Rekomendasi Metode: {best_method}**
    
    Berdasarkan analisis karakteristik setiap metode:
    
    1. **Untuk dataset besar:**
       - {'TF-IDF jika memori mencukupi' if 'TFIDF' in method_scores else 'Word2Vec untuk efisiensi'}
       - {'Word2Vec untuk keseimbangan performa' if 'WORD2VEC' in method_scores else ''}
    
    2. **Untuk akurasi tinggi:**
       - {'BERT untuk pemahaman kontekstual terbaik' if 'BERT' in method_scores else ''}
       - {'TF-IDF untuk detail fitur terbaik' if 'TFIDF' in method_scores else ''}
    
    3. **Untuk keseimbangan performa:**
       - {'Word2Vec memberikan keseimbangan terbaik' if 'WORD2VEC' in method_scores else ''}
       - Menyediakan dense vectors dengan dimensi yang cukup
       - Efisien dalam komputasi
    """)

    # Tampilkan skor evaluasi
    st.write("### 📈 Skor Evaluasi")
    score_df = pd.DataFrame([
        {"Metode": method, "Skor": score * 100} 
        for method, score in method_scores.items()
    ])
    score_df = score_df.sort_values('Skor', ascending=False)
    
    fig = px.bar(score_df, x='Metode', y='Skor',
                title='Perbandingan Skor Metode',
                labels={'Skor': 'Skor (%)'},
                color='Metode')
    st.plotly_chart(fig)

# Export semua fungsi yang diperlukan
__all__ = ['compare_preprocessing_methods', 'visualize_comparison', 
           'load_and_compare_preprocessed', 'visualize_preprocessing_comparison']