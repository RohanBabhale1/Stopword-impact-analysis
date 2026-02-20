# Results and Analysis

## 1. Overall Performance Comparison

The best-performing configuration was **Extended Stopwords + SVM**, achieving:

- **Accuracy:** 0.9123  
- **F1-score:** 0.9070  

Across all experiments, SVM consistently outperformed Logistic Regression and Naive Bayes. Stopword removal provided slight but consistent improvements over the baseline (no stopword removal), particularly when using the extended stopword list.

---

## 2. Impact of Stopword Removal

Stopword removal improved classification performance across most models:

- **Naive Bayes:** F1 improved from 0.595 to 0.635  
- **Logistic Regression:** F1 improved from 0.843 to 0.855  
- **SVM:** F1 improved from 0.906 to 0.907  

The improvement was most significant for Naive Bayes, indicating that frequency-based probabilistic models are more sensitive to high-frequency noise words.

The **Extended stopword strategy** produced the most stable improvements across models.

---

## 3. Feature Space Reduction

Feature counts across strategies:

- Baseline: 14,794 features  
- Extended Stopwords: 14,669 features  

Although the reduction (~0.8%) was relatively small, performance still improved. This suggests that removing high-frequency but low-informative words enhances class separability without significantly shrinking the vocabulary.

---

## 4. Training Time Analysis

Stopword removal improved computational efficiency:

- Naive Bayes training time decreased noticeably.
- SVM training time reduced by approximately 25%.
- Logistic Regression training time remained largely unchanged.

This indicates that reduced dimensionality can lead to faster model training, especially for linear classifiers.

---

## 5. Trade-off Analysis

Extended stopword removal achieved:

- Highest overall F1-score
- Slight feature space reduction
- Improved training efficiency
- Significant performance gains for Naive Bayes

Thus, it provides the best trade-off between predictive performance and computational cost.

---

## 6. Conclusion

1. Stopword removal positively impacts text classification performance.
2. The effect is stronger for simpler probabilistic models.
3. Extended stopwords provide the most balanced and effective strategy.
4. SVM remains the strongest classifier across all configurations.

Overall, stopword removal improves model efficiency while maintaining or slightly enhancing predictive performance.