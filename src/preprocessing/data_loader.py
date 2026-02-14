import os
import re
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

class ReutersDataLoader:
    """Load and parse Reuters-21578 SGML files"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        
    def parse_sgml_file(self, filepath):
        """Parse a single SGML file and extract articles"""
        with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()
        
        # Parse SGML
        soup = BeautifulSoup(content, 'html.parser')
        articles = []
        
        for reuters_tag in soup.find_all('reuters'):
            article = {
                'newid': reuters_tag.get('newid'),
                'topics': self._extract_topics(reuters_tag),
                'title': self._extract_text(reuters_tag, 'title'),
                'body': self._extract_text(reuters_tag, 'body'),
                'date': reuters_tag.find('date')
            }
            articles.append(article)
        
        return articles
    
    def _extract_topics(self, reuters_tag):
        """Extract topics from Reuters tag"""
        topics = reuters_tag.find('topics')
        if topics:
            return [d.text for d in topics.find_all('d')]
        return []
    
    def _extract_text(self, reuters_tag, tag_name):
        """Extract text from specific tag"""
        tag = reuters_tag.find(tag_name)
        return tag.text.strip() if tag else ""
    
    def load_all_files(self):
        """Load all SGML files from directory"""
        all_articles = []
        sgm_files = sorted(self.data_dir.glob('*.sgm'))
        
        for filepath in sgm_files:
            print(f"Processing {filepath.name}...")
            articles = self.parse_sgml_file(filepath)
            all_articles.extend(articles)
        
        return pd.DataFrame(all_articles)
    
    def filter_by_topics(self, df, min_docs=20):
        """Filter to keep only topics with sufficient documents"""
        # Explode topics list to count
        topic_counts = df.explode('topics')['topics'].value_counts()
        valid_topics = topic_counts[topic_counts >= min_docs].index
        
        # Keep only documents with valid topics
        df_filtered = df[df['topics'].apply(
            lambda x: any(t in valid_topics for t in x)
        )].copy()
        
        return df_filtered

if __name__ == "__main__":
    loader = ReutersDataLoader('data/raw')
    df = loader.load_all_files()
    print(f"Loaded {len(df)} articles")
    
    df.to_csv('data/processed/reuters_raw.csv', index=False)