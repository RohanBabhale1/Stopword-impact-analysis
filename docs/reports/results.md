# Results and Analysis

This section presents a comprehensive evaluation of static and adaptive stopword strategies on text classification performance using the Reuters-21578 dataset. The analysis includes quantitative metrics, feature space reduction, computational efficiency, and trade-off evaluation.

---

# 1. Overall Performance Comparison (Static Strategies)

The best-performing static configuration was:

**Extended Stopwords + SVM**
- Accuracy: 0.9123  
- F1-score: 0.9070  

Across all baseline experiments:

- Support Vector Machines (SVM) consistently outperformed Logistic Regression and Naive Bayes.
- Stopword removal provided measurable but modest improvements over the no-stopword baseline.
- The Extended stopword strategy produced the most stable improvements across models.

These results indicate that removing high-frequency but semantically weak tokens enhances discriminative signal without significantly altering model capacity.

---

# 2. Impact of Static Stopword Removal

## 2.1 Naive Bayes

- Baseline F1: 0.595  
- Extended F1: 0.635  

This represents a **6.72% relative improvement** in F1-score.  

Naive Bayes benefits substantially from stopword removal because it relies on term frequency distributions. High-frequency non-informative words distort probabilistic estimation, and removing them improves class-conditional probability modeling.

---

## 2.2 Logistic Regression

- Baseline F1: 0.843  
- Extended F1: 0.855  

This corresponds to a **1.42% relative improvement**.  

As a linear discriminative model, Logistic Regression is less sensitive to noisy tokens but still benefits from slight vocabulary refinement.

---

## 2.3 Support Vector Machine (SVM)

- Baseline F1: 0.906  
- Extended F1: 0.907  

The improvement is marginal (~0.11%), demonstrating that SVM is inherently robust to high-dimensional feature spaces.

---

# 3. Feature Space Reduction (Static Strategies)

| Strategy | Number of Features | Reduction (%) |
|----------|-------------------|---------------|
| None     | 14,794 | 0.00 |
| NLTK     | 14,673 | 0.82 |
| Minimal  | 14,786 | 0.05 |
| Extended | 14,669 | 0.84 |

Static stopword strategies reduce dimensionality by less than 1%.  

Despite this small reduction, measurable performance gains were observed, suggesting that even minor removal of high-frequency noise words can improve class separability.

---

# 4. Training Time Analysis (Static Strategies)

Stopword removal moderately improved computational efficiency:

- Naive Bayes: Reduced training time due to smaller token counts.
- SVM: Moderate speed improvement.
- Logistic Regression: Minimal change.

The limited dimensionality reduction explains why computational gains were relatively small under static strategies.

---

# 5. Phase 6 Innovation: Adaptive Stopword Strategy

To overcome limitations of static lists, an adaptive stopword strategy was introduced. This method dynamically identifies low-information terms using mean TF-IDF scores calculated from the training corpus.

Unlike static approaches, adaptive stopwords are dataset-specific and statistically grounded.

---

# 6. Adaptive Feature Space Reduction

The adaptive strategy reduced the feature space from:

- **14,794 → 1,836 features**
- **Feature Reduction: 87.59%**

| Stopword Strategy | Feature Reduction (%) |
|------------------|----------------------|
| None             | 0.00 |
| Minimal          | 0.05 |
| NLTK             | 0.82 |
| Extended         | 0.84 |
| **Adaptive**     | **87.59** |

This represents a dramatic dimensionality compression compared to static strategies.

The results suggest that a large proportion of vocabulary contributes minimal discriminative value for classification.

---

# 7. Adaptive Strategy: Performance Impact

## 7.1 Naive Bayes

- Extended F1: 0.635  
- Adaptive F1: 0.716  

This represents a **12.76% relative improvement** over the extended strategy.

The substantial gain confirms that Naive Bayes is highly sensitive to noisy features and benefits strongly from aggressive vocabulary pruning.

---

## 7.2 Logistic Regression

- Extended F1: 0.855  
- Adaptive F1: 0.859  

Relative improvement: **0.47%**

Although modest, this improvement occurs alongside massive dimensionality reduction, demonstrating efficient feature compression without sacrificing predictive power.

---

## 7.3 Support Vector Machine (SVM)

- Extended F1: 0.907  
- Adaptive F1: 0.901  

Relative decrease: **0.66%**

Despite a minor drop, SVM maintains competitive performance under extreme feature compression, confirming its robustness to dimensionality changes.

---

# 8. Computational Efficiency Gains

Adaptive stopwords significantly reduced training time:

- Logistic Regression: ~11.3 sec → ~2.96 sec (~73.8% reduction)
- SVM: Substantial training time reduction
- Naive Bayes: Further speed improvement

This confirms that dimensionality reduction directly impacts computational efficiency, especially for linear classifiers.

---

# 9. Performance vs Feature Compression Analysis

The adaptive strategy demonstrates that:

- Nearly 88% of the vocabulary can be removed.
- Predictive performance remains competitive.
- Naive Bayes significantly improves.
- Linear models train substantially faster.

These findings indicate diminishing returns beyond a certain vocabulary threshold. Large feature spaces may contain substantial redundancy that does not meaningfully contribute to classification performance.

---

# 10. Trade-off Analysis

| Strategy | Performance | Dimensionality | Efficiency |
|----------|------------|---------------|------------|
| Extended | Highest SVM F1 | Minimal reduction | Moderate speedup |
| Adaptive | Highest NB F1 | Massive reduction | Significant speedup |

Extended stopwords maximize SVM performance.

Adaptive stopwords provide the strongest balance between:

- Dimensionality reduction
- Computational efficiency
- Competitive predictive accuracy

---

# 11. Research Questions — Answers

## RQ1. Does stopword removal improve, degrade, or leave unchanged classification performance?

Stopword removal **improves** performance for Naive Bayes and Logistic Regression, and **leaves largely unchanged** the performance of SVM.

- Naive Bayes improved from F1 = 0.595 (baseline) to 0.716 (adaptive) — a **20.3% gain**.
- Logistic Regression improved from F1 = 0.843 (baseline) to 0.859 (adaptive) — a **1.9% gain**.
- SVM ranged only between F1 = 0.906 and 0.907 across all static strategies — a spread of just **0.11%** — confirming near-complete robustness to stopword removal.

The direction and magnitude of improvement is therefore **classifier-dependent**: generative models benefit strongly, while margin-based models are largely unaffected.

---

## RQ2. Which strategy best balances performance with feature-space economy?

The **Adaptive TF-IDF strategy** offers the best balance overall.

| Strategy | Best F1 (NB) | Best F1 (SVM) | Feature Reduction |
|---|---|---|---|
| Extended | 0.635 | 0.907 | 0.84% |
| Adaptive | 0.716 | 0.901 | 87.59% |

While Extended stopwords achieve the highest SVM F1, they compress the vocabulary by less than 1%. The Adaptive strategy achieves comparable SVM performance (only 0.66% lower), the highest NB and LR performance, and reduces the feature space by 87.59% — from 14,794 to just 1,836 features. For most real-world deployments, Adaptive provides the most efficient trade-off.

---

## RQ3. Can a data-driven adaptive strategy outperform static lists?

**Yes.** The Adaptive strategy outperforms all static strategies for both Naive Bayes and Logistic Regression.

- Over the best static strategy (Extended), Adaptive improves NB F1 by **12.76%** (0.635 → 0.716).
- LR F1 also improves slightly (0.855 → 0.859).
- Only SVM sees a minor drop of 0.66%, which is within acceptable bounds given the 87.59% compression achieved.

The Adaptive strategy succeeds because it identifies Reuters-specific low-information terms — such as *mln*, *pct*, *reuter*, *dlrs* — that appear on no standard list but add noise uniformly across all categories.

---

## RQ4. What computational efficiency gains come from aggressive vocabulary pruning?

Aggressive pruning through the Adaptive strategy produced **substantial efficiency gains**:

- **Logistic Regression**: training time reduced from ~11.35 sec to ~2.96 sec — a **73.8% speedup**.
- **SVM**: significant training time reduction due to the smaller feature matrix.
- **Naive Bayes**: further speed improvement on top of static strategy gains.
- **Feature space**: compressed from 14,794 to 1,836 features — an **87.59% reduction**.

These gains confirm that dimensionality reduction directly translates to faster training, lower memory usage, and reduced inference cost — with minimal accuracy trade-off for most classifiers.

---

# 12. Final Conclusions

1. Static stopword removal yields consistent but modest improvements.
2. Naive Bayes is highly sensitive to noisy vocabulary.
3. SVM remains robust even in high-dimensional settings.
4. Adaptive stopwords reduce feature space by 87.59%.
5. Adaptive pruning significantly improves Naive Bayes performance.
6. Logistic Regression benefits from both efficiency and slight performance gain.
7. Extreme vocabulary compression does not necessarily degrade classification performance.

Overall, adaptive stopword generation demonstrates that dataset-specific, statistically driven vocabulary pruning can substantially enhance computational efficiency while preserving or improving predictive performance.