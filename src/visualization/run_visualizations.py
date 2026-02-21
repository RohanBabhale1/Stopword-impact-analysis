import pandas as pd
from plots import ResultsVisualizer

def main():
    # Load results
    results_path = "results/tables/stopword_experiment_results.csv"
    results_df = pd.read_csv(results_path)

    # Create visualizer
    visualizer = ResultsVisualizer()

    # Generate plots
    visualizer.plot_metric_comparison(
        results_df,
        metric='f1_score',
        save_path="results/figures/f1_score_comparison.png"
    )

    visualizer.plot_metric_comparison(
        results_df,
        metric='accuracy',
        save_path="results/figures/accuracy_comparison.png"
    )

    visualizer.plot_feature_reduction(
        results_df,
        save_path="results/figures/feature_reduction.png"
    )

    visualizer.plot_performance_vs_features(
        results_df,
        save_path="results/figures/performance_vs_features.png"
    )

    visualizer.plot_heatmap(
        results_df,
        metric='f1_score',
        save_path="results/figures/f1_heatmap.png"
    )

if __name__ == "__main__":
    main()