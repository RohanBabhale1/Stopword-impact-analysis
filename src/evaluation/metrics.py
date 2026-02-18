from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import pandas as pd
import numpy as np

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true, y_pred, average='weighted'):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        return metrics
    
    def detailed_report(self, y_true, y_pred, target_names=None):
        """Generate detailed classification report"""
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        return pd.DataFrame(report).transpose()
    
    def confusion_matrix(self, y_true, y_pred):
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def compare_experiments(self, results_dict):
        """Compare multiple experimental results"""
        comparison_df = pd.DataFrame(results_dict).T
        return comparison_df