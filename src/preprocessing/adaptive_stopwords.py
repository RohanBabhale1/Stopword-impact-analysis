import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class AdaptiveStopwordHandler:
    """
    Generate adaptive stopword list based on low TF-IDF scores
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def generate(self, corpus):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        mean_scores = tfidf_matrix.mean(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        low_score_indices = np.where(mean_scores < self.threshold)[0]

        # IMPORTANT: return list (not set)
        adaptive_stopwords = list(feature_names[low_score_indices])

        return adaptive_stopwords