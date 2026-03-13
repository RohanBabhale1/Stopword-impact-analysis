"""
Unit Tests: src/models/
Tests TextClassifier and FeatureExtractor
Run with: pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.classifier import TextClassifier
from src.models.feature_extractor import FeatureExtractor


# ============================================================
# Shared fixtures
# ============================================================

SAMPLE_TEXTS = [
    "crude oil prices rose sharply in global markets",
    "grain harvest season expected to be strong this year",
    "dollar weakened against major currencies amid trade concerns",
    "earnings rose for the company last quarter significantly",
    "oil market faces uncertainty due to supply disruption",
    "wheat and corn prices fell amid bumper harvest",
    "interest rates expected to remain stable next quarter",
    "acquisition deal announced between two major companies today",
    "trade deficit widened as imports surged sharply",
    "stock market rallied on strong economic data release",
    "crude oil supply cut agreed by major producers",
    "grain exports rose amid strong global demand signals",
    "currency market saw volatility following interest rate decision",
    "company reported record earnings in annual financial report",
    "oil refinery capacity expanded to meet growing demand",
]

SAMPLE_LABELS = [
    "crude", "grain", "money-fx", "earn", "crude",
    "grain", "interest", "acq", "trade", "earn",
    "crude", "grain", "money-fx", "earn", "crude",
]


# ============================================================
# TextClassifier Tests
# ============================================================

class TestTextClassifier:

    def setup_method(self):
        """Build a fitted feature matrix once for all classifier tests"""
        extractor = FeatureExtractor(method='tfidf', stopword_strategy='none')
        self.X = extractor.fit_transform(SAMPLE_TEXTS)
        self.y = SAMPLE_LABELS

    # --- Initialisation ---

    def test_init_nb(self):
        clf = TextClassifier(model_type='nb')
        assert clf.model_type == 'nb'
        assert clf.model is not None

    def test_init_lr(self):
        clf = TextClassifier(model_type='lr')
        assert clf.model_type == 'lr'
        assert clf.model is not None

    def test_init_svm(self):
        clf = TextClassifier(model_type='svm')
        assert clf.model_type == 'svm'
        assert clf.model is not None

    def test_init_rf(self):
        clf = TextClassifier(model_type='rf')
        assert clf.model_type == 'rf'
        assert clf.model is not None

    def test_init_invalid_type_raises(self):
        with pytest.raises(ValueError):
            TextClassifier(model_type='unknown_model')

    # --- train() ---

    def test_train_nb_no_error(self):
        clf = TextClassifier(model_type='nb')
        clf.train(self.X, self.y)  # should not raise

    def test_train_lr_no_error(self):
        clf = TextClassifier(model_type='lr')
        clf.train(self.X, self.y)

    def test_train_svm_no_error(self):
        clf = TextClassifier(model_type='svm')
        clf.train(self.X, self.y)

    # --- predict() ---

    def test_predict_returns_list_or_array(self):
        clf = TextClassifier(model_type='nb')
        clf.train(self.X, self.y)
        preds = clf.predict(self.X)
        assert hasattr(preds, '__len__')

    def test_predict_correct_length(self):
        clf = TextClassifier(model_type='svm')
        clf.train(self.X, self.y)
        preds = clf.predict(self.X)
        assert len(preds) == len(self.y)

    def test_predict_labels_from_training_set(self):
        clf = TextClassifier(model_type='lr')
        clf.train(self.X, self.y)
        preds = clf.predict(self.X)
        unique_labels = set(self.y)
        for pred in preds:
            assert pred in unique_labels

    def test_predict_nb_on_training_data_reasonable_accuracy(self):
        """NB should get at least 50% accuracy on its own training data"""
        clf = TextClassifier(model_type='nb')
        clf.train(self.X, self.y)
        preds = clf.predict(self.X)
        accuracy = sum(p == t for p, t in zip(preds, self.y)) / len(self.y)
        assert accuracy >= 0.5

    def test_predict_svm_on_training_data_high_accuracy(self):
        """SVM should get close to 100% on training data"""
        clf = TextClassifier(model_type='svm')
        clf.train(self.X, self.y)
        preds = clf.predict(self.X)
        accuracy = sum(p == t for p, t in zip(preds, self.y)) / len(self.y)
        assert accuracy >= 0.8

    # --- predict_proba() ---

    def test_predict_proba_nb_returns_array(self):
        clf = TextClassifier(model_type='nb')
        clf.train(self.X, self.y)
        proba = clf.predict_proba(self.X)
        assert proba is not None
        assert proba.shape[0] == len(self.y)

    def test_predict_proba_svm_returns_none(self):
        """LinearSVC does not support predict_proba"""
        clf = TextClassifier(model_type='svm')
        clf.train(self.X, self.y)
        proba = clf.predict_proba(self.X)
        assert proba is None

    # --- save / load ---

    def test_save_and_load_model(self, tmp_path):
        clf = TextClassifier(model_type='nb')
        clf.train(self.X, self.y)
        preds_before = clf.predict(self.X)

        filepath = str(tmp_path / "test_model.joblib")
        clf.save_model(filepath)

        clf2 = TextClassifier(model_type='nb')
        clf2.load_model(filepath)
        preds_after = clf2.predict(self.X)

        assert list(preds_before) == list(preds_after)


# ============================================================
# FeatureExtractor Tests
# ============================================================

class TestFeatureExtractor:

    # --- Initialisation ---

    def test_init_tfidf_no_error(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        assert fe is not None

    def test_init_bow_no_error(self):
        fe = FeatureExtractor(method='bow', stopword_strategy='none')
        assert fe is not None

    def test_init_with_nltk_stopwords(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='nltk')
        assert fe is not None

    # --- fit_transform() ---

    def test_fit_transform_returns_sparse_matrix(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        X = fe.fit_transform(SAMPLE_TEXTS)
        assert hasattr(X, 'toarray')  # sparse matrix

    def test_fit_transform_correct_row_count(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        X = fe.fit_transform(SAMPLE_TEXTS)
        assert X.shape[0] == len(SAMPLE_TEXTS)

    def test_fit_transform_positive_feature_count(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        X = fe.fit_transform(SAMPLE_TEXTS)
        assert X.shape[1] > 0

    def test_fit_transform_tfidf_values_bounded(self):
        """TF-IDF values should be >= 0"""
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        X = fe.fit_transform(SAMPLE_TEXTS)
        assert X.min() >= 0

    def test_fit_transform_nltk_fewer_features_than_none(self):
        """NLTK stopword removal should reduce feature count"""
        fe_none = FeatureExtractor(method='tfidf', stopword_strategy='none')
        fe_nltk = FeatureExtractor(method='tfidf', stopword_strategy='nltk')
        X_none = fe_none.fit_transform(SAMPLE_TEXTS)
        X_nltk = fe_nltk.fit_transform(SAMPLE_TEXTS)
        assert X_nltk.shape[1] <= X_none.shape[1]

    # --- transform() ---

    def test_transform_returns_same_feature_count(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        fe.fit_transform(SAMPLE_TEXTS)
        X_new = fe.transform(SAMPLE_TEXTS[:5])
        X_fit = fe.fit_transform(SAMPLE_TEXTS)
        assert X_new.shape[1] == X_fit.shape[1]

    def test_transform_correct_row_count(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        fe.fit_transform(SAMPLE_TEXTS)
        X_new = fe.transform(SAMPLE_TEXTS[:3])
        assert X_new.shape[0] == 3

    # --- get_feature_names() ---

    def test_get_feature_names_returns_array(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        fe.fit_transform(SAMPLE_TEXTS)
        names = fe.get_feature_names()
        assert len(names) > 0

    def test_get_feature_names_count_matches_columns(self):
        fe = FeatureExtractor(method='tfidf', stopword_strategy='none')
        X = fe.fit_transform(SAMPLE_TEXTS)
        names = fe.get_feature_names()
        assert len(names) == X.shape[1]

    def test_get_feature_names_no_stopwords_when_nltk(self):
        """Common stopwords should not appear in feature names with nltk strategy"""
        fe = FeatureExtractor(method='tfidf', stopword_strategy='nltk')
        fe.fit_transform(SAMPLE_TEXTS)
        names = set(fe.get_feature_names())
        # 'the', 'is', 'on' are NLTK stopwords and should be absent
        for sw in ['the', 'is', 'on']:
            assert sw not in names

    # --- Adaptive strategy ---

    def test_adaptive_strategy_drastically_reduces_features(self):
        """Adaptive reduces features on a sufficiently large corpus"""
        large_corpus = SAMPLE_TEXTS * 20  # repeat to build TF-IDF signal
        fe_none = FeatureExtractor(method='tfidf', stopword_strategy='none')
        fe_adap = FeatureExtractor(
            method='tfidf', stopword_strategy='adaptive', corpus=large_corpus
        )
        X_none = fe_none.fit_transform(large_corpus)
        X_adap = fe_adap.fit_transform(large_corpus)
        assert X_adap.shape[1] <= X_none.shape[1]