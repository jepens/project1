import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def get_basic_stats(self):
        """Get basic statistics about the dataset"""
        stats = {
            'total_samples': len(self.data),
            'sentiment_distribution': self.data['sentimen'].value_counts().to_dict(),
            'avg_tweet_length': self.data['tweet'].str.len().mean(),
            'max_tweet_length': self.data['tweet'].str.len().max(),
            'min_tweet_length': self.data['tweet'].str.len().min()
        }
        return stats
    
    def plot_sentiment_distribution(self):
        """Create sentiment distribution plot"""
        fig = px.pie(
            self.data, 
            names='sentimen',
            title='Distribusi Sentiment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return fig
    
    def plot_tweet_length_distribution(self):
        """Plot tweet length distribution by sentiment"""
        self.data['tweet_length'] = self.data['tweet'].str.len()
        fig = px.box(
            self.data,
            x='sentimen',
            y='tweet_length',
            title='Distribusi Panjang Tweet berdasarkan Sentiment',
            color='sentimen'
        )
        return fig
    
    def generate_wordcloud(self, sentiment=None):
        """Generate wordcloud for specific sentiment or all data"""
        if sentiment:
            text = ' '.join(self.data[self.data['sentimen'] == sentiment]['tweet'])
        else:
            text = ' '.join(self.data['tweet'])
            
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt
    
    def get_common_terms(self, sentiment=None, n=10):
        """Get most common terms for specific sentiment or all data"""
        if sentiment:
            text = ' '.join(self.data[self.data['sentimen'] == sentiment]['tweet'])
        else:
            text = ' '.join(self.data['tweet'])
            
        words = text.split()
        return Counter(words).most_common(n)
    
    def plot_class_balance(self):
        """Plot class balance analysis"""
        class_counts = self.data['sentimen'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                text=class_counts.values,
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Distribusi Kelas Sentiment',
            xaxis_title='Sentiment',
            yaxis_title='Jumlah Sample'
        )
        return fig