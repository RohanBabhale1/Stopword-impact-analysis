# Data Directory

This directory contains the raw and processed data used in the **Stopword Impact Analysis on NLP Tasks** project.

**Author**: Babhale Rohan Laxmikant (23BCS026)  
**Course**: CS458 — Natural Language Processing  
**Supervisor**: Prof. Krishnendu Ghosh  

---

## Directory Structure

```
data/
├── raw/                  # Original Reuters-21578 SGML files (not tracked by Git)
├── processed/            # Cleaned and preprocessed data files
└── README.md             # This file
```

---

## Dataset: Reuters-21578

### Overview

The Reuters-21578 dataset is a benchmark corpus for text classification in NLP. It consists of newswire articles published by Reuters in 1987, originally collected and annotated by David Lewis at AT&T Bell Laboratories.

| Property | Value |
|---|---|
| Total articles | 21,578 |
| Articles with topic labels | 11,367 |
| Articles with body text | 19,043 |
| Unique topic categories | 120 (raw), 57 (filtered) |
| Average article length | 136 words |
| Median article length | 90 words |
| Total vocabulary (no removal) | 14,794 unique tokens |
| Stopword proportion (NLTK) | 36.74% of all tokens |

### Source

- **Official source**: [UCI KDD Archive](https://kdd.ics.uci.edu/databases/reuters21578/)
- **Original collector**: David D. Lewis, AT&T Bell Laboratories
- **Format**: 22 SGML files (`reut2-000.sgm` through `reut2-021.sgm`)
- **License**: Research use permitted; see `reuters21578.README.txt` in `raw/`

### Download Instructions

The raw SGML files are **not tracked by Git** due to licensing. To set up the data locally:

1. Download the dataset archive from the UCI KDD link above.
2. Extract all `.sgm` files into `data/raw/`.
3. Run the data loading script:

```bash
python src/preprocessing/data_loader.py
```

This will parse all SGML files and save the processed DataFrame to `data/processed/reuters_raw.csv`.

---

## Processed Data Files

After running the preprocessing pipeline, the following files are present in `data/processed/`:

| File | Description |
|---|---|
| `reuters_raw.csv` | Parsed articles from all 22 SGML files with columns: `newid`, `topics`, `title`, `body`, `date` |
| `reuters_with_analysis.csv` | Enriched dataset with additional analysis columns used in experiments (document length, stopword counts, filtered categories, etc.) |

### Column Descriptions (`reuters_raw.csv`)

| Column | Type | Description |
|---|---|---|
| `newid` | int | Unique article identifier |
| `topics` | list | List of assigned topic labels |
| `title` | str | Article headline |
| `body` | str | Full article body text |
| `date` | str | Publication date of the article |

### Column Descriptions (`reuters_with_analysis.csv`)

| Column | Type | Description |
|---|---|---|
| `newid` | int | Unique article identifier |
| `topics` | list | List of assigned topic labels |
| `title` | str | Article headline |
| `body` | str | Full article body text |
| `date` | str | Publication date of the article |
| `title_length` | int | Character count of the title |
| `title_word_count` | int | Word count of the title |
| `body_length` | int | Character count of the body |
| `body_word_count` | int | Word count of the body |
| `total_length` | int | Combined character count of title and body |
| `total_word_count` | int | Combined word count of title and body |
| `length_category` | str | Document length category (e.g., short / long), split at median word count |
| `stopword_ratio` | float | Proportion of total tokens that are NLTK stopwords |

---

## Dataset Statistics

### Topic Distribution (Top 10 Categories)

| Category | Document Count |
|---|---|
| earn | 3,987 |
| acq | 1,650 |
| money-fx | 801 |
| grain | 582 |
| crude | 578 |
| trade | 513 |
| interest | 478 |
| ship | 286 |
| wheat | 283 |
| corn | 259 |

The dataset is **heavily imbalanced** — the `earn` category alone accounts for ~38.8% of labelled documents. This is why weighted F1-score is used as the primary evaluation metric in all experiments.

### Document Length Distribution

| Statistic | Value |
|---|---|
| Mean | 136 words |
| Median | 90 words |
| Std Dev | 135 words |
| Coefficient of Variation | 99.7% |
| Short documents (≤ median) | 5,228 |
| Long documents (> median) | 5,149 |

The high CV reflects the wide variability in article length, from very short one-paragraph briefs to multi-paragraph detailed reports.

---

## Data Splits

The train/test split is an **80/20 stratified split** based on topic label distribution. This ensures proportional class representation in both sets, which is important given the severe class imbalance.

> **Note**: This project uses a stratified random split rather than the ModApte time-based split used in some historical benchmarks (Lewis et al., 2004). Results may differ slightly from those reported in older literature.

The same random seed (`random_state=42`) is used across all experiments to ensure reproducibility.

---

## Reproducibility

To reproduce the full dataset from scratch:

```bash
# Step 1: Place raw .sgm files in data/raw/

# Step 2: Parse all SGML files → generates data/processed/reuters_raw.csv
python src/preprocessing/data_loader.py

# Step 3: Run analysis enrichment → generates data/processed/reuters_with_analysis.csv
python src/preprocessing/text_cleaner.py

# Step 4: Verify file integrity
python -m pytest tests/test_preprocessing.py -v
```

---

## References

- Lewis, D. D. (1990). *Reuters-21578 Text Categorization Test Collection*. AT&T Bell Laboratories.
- Lewis, D. D., Yang, Y., Rose, T., & Li, F. (2004). RCV1: A new benchmark collection for text categorization research. *Journal of Machine Learning Research*, 5, 361–397.
