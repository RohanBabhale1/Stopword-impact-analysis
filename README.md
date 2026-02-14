# Stopword Impact Analysis on NLP Tasks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project analyzes the impact of stopword removal on various Natural Language Processing (NLP) tasks using the Reuters-21578 dataset. We investigate how different stopword removal strategies affect text classification performance across multiple machine learning models.

### Key Objectives
- Evaluate the effect of stopword removal on classification accuracy
- Compare different stopword lists (NLTK, custom, minimal, extended)
- Analyze trade-offs between performance and feature space reduction
- Develop innovative adaptive stopword strategies

## 📊 Dataset

**Reuters-21578**: A collection of 21,578 Reuters newswire articles from 1987
- **Source**: [UCI KDD Repository](https://kdd.ics.uci.edu/databases/reuters21578/)
- **Format**: SGML files
- **Categories**: 135 topics (we use top categories with ≥20 documents)
- **Task**: Multi-label text classification

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stopword-impact-analysis.git
cd stopword-impact-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Download Dataset

```bash
# Create data directory
mkdir -p data/raw

# Download Reuters-21578 dataset
cd data/raw
wget https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
tar -xzf reuters21578.tar.gz
cd ../..
```

### Run Analysis

```bash
# Load and preprocess data
python src/preprocessing/data_loader.py

# Run experiments
jupyter notebook notebooks/04_stopword_experiments.ipynb

# Or run complete pipeline
python run_experiments.py
```

## 📁 Project Structure

```
stopword-impact-analysis/
│
├── data/
│   ├── raw/                    # Original Reuters-21578 SGML files
│   └── processed/              # Cleaned and processed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_stopword_experiments.ipynb
│   └── 05_analysis_visualization.ipynb
│
├── src/
│   ├── preprocessing/          # Text cleaning and stopword handling
│   ├── models/                 # Classification models
│   ├── evaluation/             # Metrics and evaluation
│   └── visualization/          # Plotting and visualization
│
├── results/
│   ├── figures/                # Generated plots
│   ├── tables/                 # Results tables
│   └── models/                 # Saved models
│
├── docs/
│   ├── survey/                 # Literature review
│   └── reports/                # Final report and analysis
│
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Preprocessing Pipeline
1. **Text Cleaning**
   - Lowercase conversion
   - URL and email removal
   - Punctuation handling
   - Number removal (optional)

2. **Tokenization**
   - Word tokenization using NLTK
   - Optional stemming/lemmatization

3. **Stopword Strategies**
   - **None**: No stopword removal (baseline)
   - **NLTK Standard**: Default NLTK stopword list (179 words)
   - **Minimal**: Core stopwords only (10 words)
   - **Extended**: NLTK + domain-specific stopwords
   - **Adaptive**: Category-specific stopwords (innovation)

### Models Tested
- Naive Bayes (MultinomialNB)
- Logistic Regression
- Support Vector Machines (LinearSVC)
- Random Forest (optional)

### Evaluation Metrics
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)
- Feature space size
- Training time

## 📈 Key Results

### Performance Comparison

| Stopword Strategy | Model | Accuracy | F1-Score | Features |
|-------------------|-------|----------|----------|----------|
| None              | NB    | 0.XXX    | 0.XXX    | XXXXX    |
| NLTK              | NB    | 0.XXX    | 0.XXX    | XXXXX    |
| Minimal           | NB    | 0.XXX    | 0.XXX    | XXXXX    |
| Extended          | NB    | 0.XXX    | 0.XXX    | XXXXX    |
| Adaptive          | NB    | 0.XXX    | 0.XXX    | XXXXX    |

*Note: Results will be populated after running experiments*

### Key Findings
1. **Performance Impact**: Stopword removal showed [increase/decrease] of X% in F1-score
2. **Feature Reduction**: Achieved X% reduction in feature space with Y% performance trade-off
3. **Model-Specific Effects**: [Model] was most/least affected by stopword removal
4. **Category Insights**: Short news categories benefited more/less from stopword removal

## 💡 Innovation

### Adaptive Stopword Generation
We developed a novel approach to create category-specific stopword lists based on TF-IDF analysis:
- Identifies low-value terms per category
- Preserves important context-specific words
- Achieves better balance between feature reduction and performance

**Results**: Adaptive approach showed X% improvement over standard NLTK stopwords while reducing features by Y%.

## 📚 Literature Survey

Key papers reviewed:
1. Author et al. (Year) - "Title"
2. Author et al. (Year) - "Title"

Full literature review available in [`docs/survey/literature_review.md`](docs/survey/literature_review.md)

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/
```

## 📊 Visualizations

All visualizations are generated in the notebooks and saved to `results/figures/`:
- Performance comparison bar charts
- Feature reduction analysis
- Confusion matrices
- Heatmaps of results

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Babhale Rohan Laxmikant** - [GitHub Profile](https://github.com/RohanBabhale1)

## 🙏 Acknowledgments

- UCI KDD Repository for the Reuters-21578 dataset
- NLTK team for text processing tools
- Scikit-learn for machine learning implementations
- Course instructor and peers for feedback

## 📧 Contact

For questions or feedback, please reach out via:
- Email: babhale.rohan6@gmail.com
- GitHub Issues: [Create an issue](https://github.com/RohanBabhale1/stopword-impact-analysis/issues)

## 📅 Project Timeline

- Week 1: Literature survey and data exploration
- Week 2-3: Implementation and experiments
- Week 4: Innovation and analysis
- Week 5: Documentation and final report

---

**Last Updated**:14 February 2026
