from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class FeatureExtractor:
    """Extract features from text for classification"""
    
    def __init__(self, method='tfidf', stopword_strategy='none', corpus=None):
        self.method = method
        self.stopword_strategy = stopword_strategy
        
        stop_words = None
        
        # Handle stopword strategies
        if stopword_strategy == "nltk":
            from nltk.corpus import stopwords
            stop_words = stopwords.words("english")
        
        elif stopword_strategy == "adaptive":
            from src.preprocessing.adaptive_stopwords import AdaptiveStopwordHandler
            handler = AdaptiveStopwordHandler(threshold=0.0005)
            stop_words = handler.generate(corpus)
        
        # Create vectorizer
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                min_df=2,
                max_df=1.0
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                stop_words=stop_words,
                min_df=2,
                max_df=1.0
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