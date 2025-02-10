Twitter Sentiment Analysis - PILPRES 2019
Overview
Aplikasi ini adalah sistem analisis sentimen untuk mengklasifikasikan tweet terkait PILPRES 2019 ke dalam tiga kategori sentimen: Positif, Netral, dan Negatif.

## Installation

1. Clone repository:

git clone [URL_REPOSITORY]
cd [NAMA_FOLDER]

2. Buat virtual environment:
python -m venv venv

3. Aktifkan virtual environment:
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

4. Install dependencies:
pip install -r requirements.txt

## Usage Guide
1. Start Application :
streamlit run app.py

2. Navigation
Application has several main pages:

Business Understanding: Project overview and objectives
Data Analysis: Dataset exploration and visualization
Preprocessing: Text preprocessing and vectorization
Model Training: Train and tune models
Model Evaluation: Evaluate model performance
Model Management: Manage trained models
Experiment Tracking: Track and compare experiments

3. Workflow
a. Data Preprocessing

Go to "Preprocessing" page
Configure preprocessing options:

Remove mentions, URLs, hashtags
Case folding
Remove stopwords
Stemming


Select vectorization method (TF-IDF)
Run preprocessing

b. Model Training

Go to "Model Training" page
Select model type:

Random Forest
LSTM
DistilBERT


Configure model parameters
Set training options:

Enable/disable hyperparameter optimization
Set validation split
Name experiment


# Start training

c. Model Evaluation

Go to "Model Evaluation" page
Select trained model
View evaluation metrics:

Accuracy, Precision, Recall, F1-Score
Confusion Matrix
ROC Curves



d. Experiment Tracking

Go to "Experiment Tracking" page
View experiment history
Compare model performances
Analyze training trends

## Features

Text preprocessing with multiple options
Multiple model implementations
Hyperparameter optimization with Optuna
Comprehensive model evaluation
Experiment tracking and comparison
Interactive visualizations
Model versioning and management

Models
1. Random Forest

Feature importance analysis
Grid search optimization
Macro-average metrics

2. LSTM

Bidirectional option
Multiple layer support
Dropout regularization

3. DistilBERT

Pre-trained Indonesian BERT
Fine-tuning capabilities
Transfer learning
 
# Contributing

Fork repository
Create feature branch
Commit changes
Push to branch
Create Pull Request

# Warning
Aplikasi ini menggunakan data yang sensitif dengan sentimen politik. Gunakan dengan bijak dan bertanggung jawab.
Notes

Pastikan dataset tersedia di folder data/
Model yang sudah dilatih akan tersimpan di folder experiments/models/
Experiment tracking disimpan di experiments/metadata/

# Support
Jika ada pertanyaan atau masalah, silakan buka issue di repository GitHub.