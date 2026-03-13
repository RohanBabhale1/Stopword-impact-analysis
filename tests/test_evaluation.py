"""
Unit Tests: src/evaluation/metrics.py
Tests ModelEvaluator
Run with: pytest tests/test_evaluation.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import ModelEvaluator


# ============================================================
# Shared test data
# ============================================================

Y_TRUE_BINARY  = ['cat', 'cat', 'dog', 'dog', 'cat', 'dog']
Y_PRED_PERFECT = ['cat', 'cat', 'dog', 'dog', 'cat', 'dog']
Y_PRED_PARTIAL = ['cat', 'dog', 'dog', 'cat', 'cat', 'dog']  # 2 wrong
Y_PRED_ALL_WRONG = ['dog', 'dog', 'cat', 'cat', 'dog', 'cat']

Y_TRUE_MULTI  = ['earn', 'crude', 'grain', 'earn', 'crude', 'grain', 'acq', 'acq']
Y_PRED_MULTI  = ['earn', 'crude', 'earn',  'earn', 'crude', 'grain', 'acq', 'earn']


# ============================================================
# ModelEvaluator Tests
# ============================================================

class TestModelEvaluator:

    def setup_method(self):
        self.evaluator = ModelEvaluator()

    # --- evaluate() return structure ---

    def test_evaluate_returns_dict(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert isinstance(result, dict)

    def test_evaluate_has_accuracy_key(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert 'accuracy' in result

    def test_evaluate_has_precision_key(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert 'precision' in result

    def test_evaluate_has_recall_key(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert 'recall' in result

    def test_evaluate_has_f1_score_key(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert 'f1_score' in result

    # --- Perfect predictions ---

    def test_evaluate_perfect_accuracy_is_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert result['accuracy'] == pytest.approx(1.0)

    def test_evaluate_perfect_f1_is_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert result['f1_score'] == pytest.approx(1.0)

    def test_evaluate_perfect_precision_is_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert result['precision'] == pytest.approx(1.0)

    def test_evaluate_perfect_recall_is_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert result['recall'] == pytest.approx(1.0)

    # --- Partial predictions ---

    def test_evaluate_partial_accuracy_between_0_and_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PARTIAL)
        assert 0.0 < result['accuracy'] < 1.0

    def test_evaluate_partial_accuracy_correct_value(self):
        # 4 correct out of 6
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PARTIAL)
        assert result['accuracy'] == pytest.approx(4/6, rel=1e-3)

    def test_evaluate_partial_f1_less_than_perfect(self):
        perfect = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PERFECT)
        partial = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PARTIAL)
        assert partial['f1_score'] < perfect['f1_score']

    # --- Metrics are bounded ---

    def test_evaluate_all_metrics_between_0_and_1(self):
        result = self.evaluator.evaluate(Y_TRUE_BINARY, Y_PRED_PARTIAL)
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 0.0 <= result[key] <= 1.0

    def test_evaluate_multiclass_metrics_bounded(self):
        result = self.evaluator.evaluate(Y_TRUE_MULTI, Y_PRED_MULTI)
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 0.0 <= result[key] <= 1.0

    # --- Zero-division safety ---

    def test_evaluate_no_crash_on_unseen_class(self):
        """Should not raise even if a class appears only in y_true"""
        y_true = ['earn', 'crude', 'grain']
        y_pred = ['earn', 'earn',  'earn']
        result = self.evaluator.evaluate(y_true, y_pred)
        assert isinstance(result, dict)

    # --- Multiclass ---

    def test_evaluate_multiclass_returns_dict(self):
        result = self.evaluator.evaluate(Y_TRUE_MULTI, Y_PRED_MULTI)
        assert isinstance(result, dict)

    def test_evaluate_multiclass_accuracy_correct(self):
        # earn→earn✓, crude→crude✓, grain→earn✗, earn→earn✓,
        # crude→crude✓, grain→grain✓, acq→acq✓, acq→earn✗  → 6/8
        result = self.evaluator.evaluate(Y_TRUE_MULTI, Y_PRED_MULTI)
        assert result['accuracy'] == pytest.approx(6/8, rel=1e-3)

    # --- detailed_report() ---

    def test_detailed_report_returns_dataframe(self):
        result = self.evaluator.detailed_report(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert isinstance(result, pd.DataFrame)

    def test_detailed_report_contains_classes(self):
        result = self.evaluator.detailed_report(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert 'cat' in result.index or 'cat' in result.columns

    def test_detailed_report_with_target_names(self):
        result = self.evaluator.detailed_report(
            Y_TRUE_BINARY, Y_PRED_PERFECT,
            target_names=['cat', 'dog']
        )
        assert isinstance(result, pd.DataFrame)

    # --- confusion_matrix() ---

    def test_confusion_matrix_returns_array(self):
        cm = self.evaluator.confusion_matrix(Y_TRUE_BINARY, Y_PRED_PERFECT)
        assert hasattr(cm, '__len__')

    def test_confusion_matrix_shape_matches_classes(self):
        cm = self.evaluator.confusion_matrix(Y_TRUE_BINARY, Y_PRED_PERFECT)
        n_classes = len(set(Y_TRUE_BINARY))
        assert cm.shape == (n_classes, n_classes)

    def test_confusion_matrix_perfect_is_diagonal(self):
        cm = self.evaluator.confusion_matrix(Y_TRUE_BINARY, Y_PRED_PERFECT)
        # Off-diagonal should all be zero for perfect predictions
        np.fill_diagonal(cm, 0)
        assert cm.sum() == 0

    def test_confusion_matrix_sum_equals_n_samples(self):
        cm = self.evaluator.confusion_matrix(Y_TRUE_BINARY, Y_PRED_PARTIAL)
        assert cm.sum() == len(Y_TRUE_BINARY)

    def test_confusion_matrix_non_negative(self):
        cm = self.evaluator.confusion_matrix(Y_TRUE_MULTI, Y_PRED_MULTI)
        assert (cm >= 0).all()

    # --- compare_experiments() ---

    def test_compare_experiments_returns_dataframe(self):
        results = {
            'exp1': {'accuracy': 0.85, 'f1_score': 0.83},
            'exp2': {'accuracy': 0.90, 'f1_score': 0.89},
        }
        df = self.evaluator.compare_experiments(results)
        assert isinstance(df, pd.DataFrame)

    def test_compare_experiments_correct_shape(self):
        results = {
            'exp1': {'accuracy': 0.85, 'f1_score': 0.83},
            'exp2': {'accuracy': 0.90, 'f1_score': 0.89},
            'exp3': {'accuracy': 0.91, 'f1_score': 0.90},
        }
        df = self.evaluator.compare_experiments(results)
        assert df.shape[0] == 3

    def test_compare_experiments_columns_are_metrics(self):
        results = {
            'nb_none': {'accuracy': 0.68, 'f1_score': 0.60},
            'svm_extended': {'accuracy': 0.91, 'f1_score': 0.91},
        }
        df = self.evaluator.compare_experiments(results)
        assert 'accuracy' in df.columns
        assert 'f1_score' in df.columns