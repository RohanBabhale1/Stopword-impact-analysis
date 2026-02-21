import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ResultsVisualizer:
    """Visualize experimental results"""
    
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_metric_comparison(self, results_df, metric='f1_score', 
                               save_path=None):
        """Compare metrics across experiments"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_df = results_df.pivot(
            index='stopword_strategy', 
            columns='model_type', 
            values=metric
        )
        
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel('Stopword Strategy')
        ax.legend(title='Model Type')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_reduction(self, results_df, save_path=None):
        """Visualize feature space reduction"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in results_df['model_type'].unique():
            model_data = results_df[results_df['model_type'] == model]
            ax.plot(
                model_data['stopword_strategy'], 
                model_data['num_features'],
                marker='o', label=model
            )
        
        ax.set_title('Feature Space Reduction by Stopword Strategy')
        ax.set_ylabel('Number of Features')
        ax.set_xlabel('Stopword Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_vs_features(self, results_df, save_path=None):
        """Plot performance vs. number of features"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in results_df['stopword_strategy'].unique():
            strategy_data = results_df[results_df['stopword_strategy'] == strategy]
            ax.scatter(
                strategy_data['num_features'],
                strategy_data['f1_score'],
                label=strategy,
                s=100,
                alpha=0.6
            )
        
        ax.set_title('F1-Score vs. Feature Space Size')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('F1-Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heatmap(self, results_df, metric='f1_score', save_path=None):
        """Create heatmap of results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot_df = results_df.pivot(
            index='stopword_strategy',
            columns='model_type',
            values=metric
        )
        
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.4f', 
            cmap='YlGnBu',
            ax=ax,
            cbar_kws={'label': metric}
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()