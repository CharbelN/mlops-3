# mlops-2025
Repo for MLOps Course of class 2025-2026

## Setup

### 1. Fork and Clone Repository
- Forked the [mlops-2025 repository](https://github.com/najielhachem/mlops-2025/) to GitHub account
- Cloned the fork locally:
  ```bash
  git clone https://github.com/<your-username>/mlops-2025.git
  cd mlops-2025
  ```

### 2. Configure Upstream Remote
- Added upstream remote to track the instructor's repository:
  ```bash
  git remote add upstream https://github.com/najielhachem/mlops-2025.git
  git fetch upstream
  ```

✅ Repository setup complete!

## 2. Package Creation

**Goal**: Get a minimal, runnable package with uv.

### Package Structure
The package is already initialized with the following structure:
- **src/ layout**: Package code is organized in `src/mlops_2025/`
- **pyproject.toml**: Project configuration with dependencies
- **Python version**: >=3.12

### Dependencies
All notebook dependencies are configured in `pyproject.toml`:

**Main dependencies** (from titanic-survival.ipynb):
- `numpy>=2.3.3` - Numerical computing
- `pandas>=2.3.3` - Data manipulation and analysis
- `matplotlib>=3.10.6` - Data visualization
- `seaborn>=0.13.2` - Statistical data visualization
- `scikit-learn>=1.7.2` - Machine learning library

**Dev dependencies**:
- `jupyter>=1.1.1` - Jupyter notebook environment

### Branch
```bash
git checkout -b setup_package
```

✅ Package setup complete with all required dependencies!

## 3. Notebook → Scripts

Converting the Titanic survival prediction notebook into modular CLI scripts.

### 3.1 Preprocessing Script

**File**: `scripts/preprocess.py`

**Purpose**: Load raw CSV data, perform initial cleaning, and save processed data.

**Features**:
- Drop Cabin column (too many missing values)
- Fill missing Embarked values with 'S'
- Fill missing Fare values with mean
- Fill missing Age values using group median (by Sex and Pclass)
- Convert Survived column to int64

**Usage**:
```bash
# Single file mode
uv run python scripts/preprocess.py \
  --input data/raw/train.csv \
  --output data/processed/train_processed.csv

# Combined train/test mode
uv run python scripts/preprocess.py \
  --input data/raw/train.csv \
  --test_path data/raw/test.csv \
  --output data/processed/train_processed.csv \
  --output_test data/processed/test_processed.csv
```

**Branch**: `feature/preprocess`
