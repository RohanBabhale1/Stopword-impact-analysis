# Methodology

This section describes the experimental setup, preprocessing techniques, stopword strategies, feature extraction methods, classification models, evaluation metrics, and the proposed adaptive stopword innovation.

The primary objective of this study was to systematically evaluate the impact of different stopword removal strategies on text classification performance under controlled experimental conditions.

---

## 1. Experimental Framework

The experimental pipeline followed a structured and consistent process:

1. Data preprocessing  
2. Stopword strategy application  
3. TF-IDF feature extraction  
4. Model training  
5. Performance evaluation  
6. Feature reduction and computational analysis  

All experiments were conducted using identical train-test splits and evaluation metrics to ensure fair comparison across stopword strategies.

---

## 2. Data Preparation and Preprocessing

The dataset consisted of textual documents used for supervised classification.

The following preprocessing steps were applied:

- Conversion to lowercase
- Removal of punctuation and special characters
- Tokenization
- Handling of missing values
- Removal of extra whitespace

### 2.1 Document Length Analysis

Document length was calculated as the number of tokens per document.

The dataset was split using the median document length:

- Short documents: 5,228  
- Long documents: 5,149  

This balanced distribution ensured that experiments were not biased toward a specific document size group.

---

## 3. Stopword Strategies

Five stopword configurations were evaluated.

### 3.1 None (Baseline)
No stopword removal was applied. All terms were retained in the vocabulary.

### 3.2 NLTK Stopwords
A predefined English stopword list was used to remove common high-frequency words.

### 3.3 Minimal Stopwords
A small manually curated stopword list was applied, containing only the most obvious non-informative words.

### 3.4 Extended Stopwords
An expanded static stopword list was created by extending the standard list with additional high-frequency dataset-specific terms.

### 3.5 Adaptive Stopwords (Proposed Innovation)
A dynamic, data-driven stopword list was generated using statistical analysis of the training corpus.

This approach identifies low-information words automatically rather than relying on predefined lists.

---

## 4. Adaptive Stopword Generation Procedure

The adaptive strategy was implemented using the following steps:

1. Compute TF-IDF matrix on the training corpus.
2. Calculate the mean TF-IDF score for each term across all training documents.
3. Identify terms with mean TF-IDF score below a threshold value (0.001).
4. Treat these low-scoring terms as adaptive stopwords.
5. Reapply TF-IDF vectorization after removing the identified stopwords.

Important considerations:

- Stopwords were generated using training data only to prevent data leakage.
- The threshold value (0.0005) was selected empirically to balance vocabulary compression and information preservation.
- The adaptive strategy dynamically adjusts to dataset characteristics.

---

## 5. Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) was used for feature representation.

Configuration details:

- Minimum document frequency: 2  
- No maximum document frequency restriction  
- Stopword strategy applied per experiment  

The number of extracted features was recorded for each configuration to analyze dimensionality differences.

---

## 6. Classification Models

Three machine learning classifiers were evaluated:

1. Naive Bayes  
2. Logistic Regression  
3. Support Vector Machine (SVM)  

Each model was trained using the same training data and evaluated on the same test set to maintain experimental consistency.

---

## 7. Evaluation Metrics

Model performance was assessed using:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  
- Number of extracted features  
- Training time (in seconds)  

This allowed evaluation of both predictive performance and computational efficiency.

---

## 8. Feature Reduction Analysis

Feature reduction percentage was computed using:

Feature Reduction (%) =  
((Baseline Features − Strategy Features) / Baseline Features) × 100

This metric quantifies how effectively each stopword strategy reduces vocabulary size relative to the baseline.

---

## 9. Computational Efficiency Assessment

Training time was recorded for each model configuration.

The impact of stopword removal on:

- Dimensionality of feature space  
- Training speed  
- Model complexity  

was systematically analyzed.

Dimensionality reduction directly influences computational cost, especially for linear classifiers.

---

## 10. Experimental Validity and Fairness

To ensure methodological rigor:

- Identical train-test splits were used across all experiments.
- TF-IDF configuration remained constant.
- Evaluation metrics were consistent across strategies.
- Adaptive stopwords were generated exclusively from training data.

These controls ensure that observed performance differences are attributable solely to stopword strategy variations.

---

## Methodological Summary

This study employs a controlled comparative framework to analyze the effect of static and adaptive stopword strategies on text classification.

The methodology enables evaluation of:

- Predictive performance
- Vocabulary dimensionality
- Computational efficiency
- Trade-offs between accuracy and model complexity

The adaptive stopword strategy introduces a statistically grounded, dataset-specific vocabulary pruning mechanism within a standardized experimental pipeline.