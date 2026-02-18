from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class FeatureExtractor:
    """Extract features from text for classification"""
    
    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.8
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.8
            )
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()