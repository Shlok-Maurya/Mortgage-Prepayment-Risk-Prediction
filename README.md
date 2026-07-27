# 🏦 Mortgage Prepayment Risk Prediction

**Predicting whether a mortgage borrower will prepay their loan using Machine Learning on Fannie Mae Single-Family Loan Performance Data (2000–2020).**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Pipeline Phases](#pipeline-phases)
- [Key Features Engineered](#key-features-engineered)
- [Models & Results](#models--results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Team](#team)
- [License](#license)

---

## Overview

Mortgage prepayment is a significant risk factor for financial institutions. When borrowers pay off their mortgages early (typically through refinancing), it disrupts expected cash flows and impacts profitability. This project builds a machine learning pipeline to predict prepayment risk using **20 years of Fannie Mae loan-level data** spanning five critical economic eras:

| Cohort | Economic Era | Characteristics |
|--------|-------------|-----------------|
| **2000 Q1** | Dot-Com Boom | High interest rates |
| **2005 Q1** | Housing Bubble Peak | Subprime lending surge |
| **2008 Q1** | Financial Crisis | Market crash |
| **2013 Q1** | Post-Crisis Recovery | Strict lending standards |
| **2017 Q1** | Pre-COVID Stability | Used as out-of-time test set |
| **2020 Q1** | COVID-19 Pandemic | Record low rates |

---

## Problem Statement

> Given a mortgage loan's characteristics (credit score, debt-to-income ratio, loan purpose, property type, geographic state, etc.), can we predict whether the borrower will **prepay** the loan before maturity?

This is a **binary classification** problem:
- **Class 0**: Active (loan continues as scheduled)
- **Class 1**: Prepaid (borrower pays off the loan early)

---

## Dataset

The raw dataset is hosted on **Hugging Face** due to its massive size (~48 GB total):

🔗 **[Shlok112003/Fannie_Mae_Single-Family_Loan_Performance](https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance)**

### Raw Data
| File | Period | Size |
|------|--------|------|
| `2000Q1.csv` | Q1 2000 | ~2.3 GB |
| `2005Q1.csv` | Q1 2005 | ~6.5 GB |
| `2008Q1.csv` | Q1 2008 | ~5.9 GB |
| `2013Q1.csv` | Q1 2013 | ~17.2 GB |
| `2017Q1.csv` | Q1 2017 | ~8.4 GB |
| `2020Q1.csv` | Q1 2020 | ~8.2 GB |

### Processed Data
| File | Description | Size |
|------|-------------|------|
| `train_ready.csv` | Balanced training set (2000, 2005, 2008, 2013) | ~80 MB |
| `test_ready.csv` | Natural distribution test set (2017) | ~75 MB |

### Data Source
[Fannie Mae Single-Family Loan Performance Data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)

---

## Project Architecture

```
Raw Data (48 GB, Pipe-delimited)
        │
        ▼
┌─────────────────────────┐
│  Phase 1: Verification  │  Validate file integrity, column counts, delimiters
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│    Phase 2: EDA         │  20-year narrative across 5 economic eras
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Phase 3: ETL Pipeline  │  N-1 Logic → Balancing → Feature Engineering → Encoding
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Phase 4: ML Showdown   │  Logistic Regression vs. XGBoost
└─────────────────────────┘
```

---

## Work Done / Key Contributions

This project involved end-to-end data mining and machine learning — from raw data acquisition to model explainability. Here is a summary of everything that was accomplished:

### 1. Large-Scale Data Acquisition & Management
- Acquired **~48 GB** of raw Fannie Mae Single-Family Loan Performance data across **6 quarterly cohorts** (2000Q1, 2005Q1, 2008Q1, 2013Q1, 2017Q1, 2020Q1)
- Each raw file contains **110 pipe-delimited columns** representing monthly loan performance records
- Hosted the full dataset on [Hugging Face](https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance) for reproducibility and easy access

### 2. Data Verification (Phase 1)
- Built a verification protocol that checks file existence, file sizes, delimiters, and column counts for all raw files
- Confirmed all **5 primary cohort files** are valid with 110 columns each (totaling over 2.2 million unique loans)

### 3. Exploratory Data Analysis — The 20-Year Narrative (Phase 2)
- Loaded and processed **2,296,411 unique loan terminal states** across all 5 cohorts for analysis
- Created **5 key visualizations** that tell the macroeconomic story of mortgage prepayment:
  - **Graph 1: The Chronological Imbalance** — Mapped prepayment vs. active rates across all 5 economic eras, proving that prepayment probability is *not static* but entirely dependent on the economic year
  - **Graph 2: The Catalyst (Interest Rate Evolution)** — Tracked origination rate distributions across 20 years, showing how rate drops drive prepayment surges
  - **Graph 3: The Risk Profile (FICO Evolution)** — Analyzed how credit score distributions shifted post-2008 crisis (tighter lending standards)
  - **Graph 4: Borrower Leverage (DTI Evolution)** — Examined debt-to-income ratio trends across economic cycles
  - **Graph 5: Mathematical Foundation (Feature Correlation Heatmap)** — Computed the full correlation matrix across all 20 years of combined data with proper NaN imputation

### 4. ETL Pipeline with Anti-Leakage Design (Phase 3)
- Built a robust ETL pipeline processing **~48 GB of raw data** into ML-ready datasets
- Implemented **N-1 Logic (Look-Ahead Bias Prevention)**:
  - For **Prepaid loans**: extracts the *second-to-last* (N-1) record to avoid using future information
  - For **Active loans**: extracts the *final* (N) record as their outcome is still in progress
- **Per-Cohort Balancing**: undersampled majority class to exactly 50/50 *within each training cohort* to preserve economic timeline integrity (rather than blending all years together)
- **Feature Engineering**:
  - Calculated dynamic `REFINANCE_INCENTIVE` = `ORIG_RATE - MARKET_RATE` using year-specific market rates
  - Binned credit scores into 5 buckets: Very Poor → Excellent (using quantile-based binning)
  - Binned DTI into 3 buckets: Low Debt → High Debt
  - Extracted seasonality features (year, month)
  - Robust boolean parsing for first-time homebuyer flag
- **One-Hot Encoding & Column Alignment**: encoded all categorical features (PURPOSE, PROP_TYP, OCC_STAT, STATE, CSCORE_BUCKET, DTI_BUCKET) and perfectly aligned training and testing column sets to prevent model crashes on unseen variables
- **Temporal Split (Out-of-Time Validation)**:
  - **Training Data**: 2000Q1, 2005Q1, 2008Q1, 2013Q1 → **512,744 samples** (balanced)
  - **Testing Data**: 2017Q1 → **487,789 samples** (natural imbalanced distribution: 351,247 prepaid / 136,542 active)

### 5. Model Training & Evaluation (Phase 4)
- Trained **two contender models**:
  - **Logistic Regression** (traditional banking baseline) — with StandardScaler preprocessing
  - **XGBoost** (advanced tree-based model) — with tuned hyperparameters
- Evaluated using **multiple metrics** suited for imbalanced classification:
  - ROC-AUC, PR-AUC, F1-Score, Precision, Recall
- Generated **Confusion Matrix visualizations** for both models
- Performed **SHAP (SHapley Additive exPlanations) analysis** for model explainability — identifying which features most influence prepayment predictions
- **Final result**: XGBoost outperformed Logistic Regression across all metrics on the out-of-time 2017 test set

### 6. Total Scale of Data Processed
| Metric | Value |
|--------|-------|
| Raw data size | ~48 GB |
| Raw columns per file | 110 |
| Total unique loans analyzed (EDA) | 2,296,411 |
| Training samples (after balancing) | 512,744 |
| Testing samples (natural distribution) | 487,789 |
| Final engineered features | 72 |
| Cohorts spanning | 20 years (2000–2020) |

---

## Pipeline Phases

### Phase 1 — Raw Data Verification (`phase1.ipynb`)
- Validates existence and integrity of all raw CSV files
- Checks file sizes, delimiters (pipe `|`), and column counts (110 columns expected)

### Phase 2 — Exploratory Data Analysis (`eda.ipynb`)
- Analyzes borrower behavior across 5 economic eras (2000–2020)
- Visualizes credit score distributions, DTI ratios, prepayment rates
- Explores correlations and macroeconomic patterns

### Phase 3 — ETL Pipeline (`etl_pipeline.ipynb`)
- **Extraction**: Applies strict **N-1 Logic** to prevent look-ahead bias — uses the *second-to-last* record for prepaid loans and the *final* record for active loans
- **Per-Cohort Balancing**: Undersamples majority class to 50/50 *within each training cohort* to preserve economic timeline
- **Feature Engineering**: Calculates `REFINANCE_INCENTIVE`, bins credit scores and DTI into buckets, handles seasonality
- **Encoding & Alignment**: One-Hot Encoding of categorical features; aligns train/test column sets
- **Temporal Split (Out-of-Time Validation)**:
  - **Train** (The Past): 2000, 2005, 2008, 2013 → 512,744 samples
  - **Test** (The Future): 2017 → 487,789 samples (natural imbalanced distribution)

### Phase 4 — Model Training & Evaluation (`modeltraining.ipynb`)
- Trains Logistic Regression (baseline) and XGBoost (champion)
- Evaluates using ROC-AUC, PR-AUC, F1-Score, Precision, Recall
- SHAP-based explainability analysis
- Confusion matrix visualizations

---

## Key Features Engineered

| Feature | Description |
|---------|-------------|
| `REFINANCE_INCENTIVE` | `ORIG_RATE - MARKET_RATE` (dynamic, year-based) |
| `CSCORE_BUCKET` | Credit score binned into 5 categories (Very Poor → Excellent) |
| `DTI_BUCKET` | Debt-to-income binned into 3 categories (Low → High Debt) |
| `FTHB_FLG` | First-time home buyer flag (binary) |
| `PURPOSE` | Loan purpose (Purchase/Refinance/Unknown) — One-Hot Encoded |
| `PROP_TYP` | Property type — One-Hot Encoded |
| `OCC_STAT` | Occupancy status — One-Hot Encoded |
| `STATE` | U.S. state — One-Hot Encoded |

**Total features after encoding**: 72

---

## Models & Results

### Out-of-Time Validation Scoreboard (Test on 2017 data)

| Metric | Logistic Regression | XGBoost (Tuned) |
|--------|:------------------:|:---------------:|
| **ROC-AUC** | 0.5349 | **0.5999** |
| **PR-AUC** | 0.7370 | **0.7753** |
| **F1-Score** | — | **0.7408** |
| **Precision** | — | **0.7623** |
| **Recall** | — | **0.7204** |

> 🏆 **XGBoost** outperforms the traditional banking baseline across all metrics.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10 |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Imbalanced Data** | imbalanced-learn |
| **Model Persistence** | joblib |
| **Dashboard** | Streamlit |

---

## Getting Started

### Prerequisites
- Python 3.10+
- ~2 GB free disk space (for processed data)

### Installation

```bash
# Clone the repository
git clone https://github.com/Shlok-Maurya/Mortgage-Prepayment-Risk-Prediction.git
cd Mortgage-Prepayment-Risk-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Download the Data

The processed datasets can be loaded directly from Hugging Face:

```python
import pandas as pd

# Load processed data
train = pd.read_csv("https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance/resolve/main/processed/train_ready.csv")
test = pd.read_csv("https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance/resolve/main/processed/test_ready.csv")
```

Or download raw files from the [Hugging Face dataset page](https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance) and place them in `data/`.

### Run the Notebooks

Execute the notebooks in order:
1. `new_notebooks/phase1.ipynb` — Data verification
2. `new_notebooks/eda.ipynb` — Exploratory analysis
3. `new_notebooks/etl_pipeline.ipynb` — ETL & feature engineering
4. `new_notebooks/modeltraining.ipynb` — Model training & evaluation

---

## Project Structure

```
Mortgage-Prepayment-Risk-Prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── data/                          # Data directory (git-ignored)
│   ├── 2000Q1.csv                 # Raw Fannie Mae files
│   ├── 2005Q1.csv
│   ├── 2008Q1.csv
│   ├── 2013Q1.csv
│   ├── 2017Q1.csv
│   ├── 2020Q1.csv
│   └── processed/
│       ├── train_ready.csv        # ML-ready training data
│       └── test_ready.csv         # ML-ready testing data
└── new_notebooks/
    ├── phase1.ipynb               # Raw data verification
    ├── eda.ipynb                   # Exploratory data analysis
    ├── etl_pipeline.ipynb          # ETL & feature engineering
    └── modeltraining.ipynb         # Model training & evaluation
```

> **Note**: Raw and processed data files are hosted on [Hugging Face](https://huggingface.co/datasets/Shlok112003/Fannie_Mae_Single-Family_Loan_Performance) and excluded from this repository via `.gitignore`.

---

## License

This project is for academic and educational purposes.
Data sourced from [Fannie Mae](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data).
