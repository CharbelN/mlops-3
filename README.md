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

### 3.2 Feature Engineering Script

**File**: `scripts/featurize.py`

**Purpose**: Extract features from preprocessed data.

**Features**:
- **Extract Title**: Parse title from Name column (Mr, Mrs, Miss, Master, Rare)
  - Groups titles
  - Normalizes: Mlle → Miss, Ms → Miss, Mme → Mrs
- **Create Family_size**: Combines SibSp + Parch + 1, then categorizes:
  - "Alone" (1 person)
  - "Small" (2-4 people)
  - "Large" (5+ people)
- **Drop unnecessary columns**: Name, Ticket, Parch, SibSp
- **Convert Age to int64**

**Usage**:
```bash
uv run python scripts/featurize.py \
  --input data/processed/train_processed.csv \
  --output data/featurized/train_featurized.csv
```

**Branch**: `feature/featurize`

### 3.3 Training Script

**File**: `scripts/train.py`

**Purpose**: Train a baseline model and save it for later use.

**Features**:
- **Preprocessing pipeline**: 
  - MinMaxScaler for numeric features (Age, Fare)
  - OneHotEncoder for categorical features (Sex, Embarked, Title, Family_size)
  - OrdinalEncoder for ordinal features (Pclass)
- **Baseline model**: LogisticRegression with max_iter=1000
- **Cross-validation**: 5-fold CV to evaluate model performance
- **Model persistence**: Save trained pipeline using pickle

**Usage**:
```bash
uv run python scripts/train.py \
  --input data/featurized/train_featurized.csv \
  --output models/model.pkl
```

**Output**:
- Trained model saved as pickle file in `models/` directory
- Reports CV accuracy and training accuracy

**Branch**: `feature/train`

### 3.4 Evaluation Script

**File**: `scripts/evaluate.py`

**Purpose**: Evaluate a trained model on test/validation data.

**Features**:
- Load trained model from pickle file
- Compute accuracy on evaluation dataset
- Generate confusion matrix with interpretation
- Display classification report (precision, recall, F1-score)
- Optional: Save metrics to JSON file

**Usage**:
```bash
uv run python scripts/evaluate.py \
  --model models/model.pkl \
  --input data/featurized/test_featurized.csv

uv run python scripts/evaluate.py \
  --model models/model.pkl \
  --input data/featurized/test_featurized.csv \
  --output results/metrics.json
```

**Output**:
- Accuracy score
- Confusion matrix (TP, TN, FP, FN)
- Classification report
- Optional: JSON file with metrics

**Branch**: `feature/evaluate`

### 3.5 Prediction/Inference Script

**File**: `scripts/predict.py`

**Purpose**: Make predictions on unlabeled data using a trained model.

**Features**:
- Load trained model from pickle file
- Process featurized data without labels
- Generate predictions (0 = Not Survived, 1 = Survived)
- Include prediction probabilities for both classes
- Save predictions to CSV with PassengerId
- Display prediction statistics

**Usage**:
```bash
uv run python scripts/predict.py \
  --model models/model.pkl \
  --input data/featurized/test_featurized.csv \
  --output predictions/predictions.csv
```

**Output CSV columns**:
- `PassengerId`: Original passenger ID (if available)
- `Survived`: Predicted class (0 or 1)
- `Probability_NotSurvived`: Probability of class 0
- `Probability_Survived`: Probability of class 1

**Branch**: `feature/predict`

---

## 4. Running the Pipeline

### Automated Pipeline Execution

Use the provided scripts to run the complete pipeline:

**Bash (Linux/Mac/Git Bash):**
```bash
chmod +x run_pipeline.sh

./run_pipeline.sh
```

**PowerShell (Windows):**
```powershell
.\run_pipeline.ps1
```

The pipeline scripts automatically execute all 5 steps in sequence:
1. Preprocess raw data
2. Engineer features
3. Train model
4. Evaluate model (if labels available)
5. Make predictions

**Branch**: `feature/run-pipeline`
---

## 5. Class-Based Architecture

**Goal**: Refactor scripts into class-based components with interfaces for better modularity and extensibility.

### 5.1 Preprocessing Classes

**Branch**: `feature/preprocessing-class`

**Module**: `src/mlops_2025/preprocessing/`

**Architecture**:
- **BasePreprocessor**: Abstract base class defining the preprocessing interface
- **Preprocessor**: Concrete implementation for single-file preprocessing
- **CombinedPreprocessor**: Implementation for combined train/test preprocessing

**Key Benefits**:
- Clean separation of concerns
- Easy to swap implementations
- Testable components
- Reusable across different scripts

**Class Structure**:
```python
from mlops_2025.preprocessing import Preprocessor, CombinedPreprocessor

# Single file preprocessing
preprocessor = Preprocessor()
df_processed = preprocessor.process(df)

# Combined train/test preprocessing
combined_preprocessor = CombinedPreprocessor()
df_processed = combined_preprocessor.process(df_combined)
```

**Files**:
- `base_preprocessor.py` - Abstract interface with `process()` method
- `preprocessor.py` - Concrete implementations
- `__init__.py` - Module exports

**Updated Script**: `scripts/preprocess.py` now uses these classes instead of standalone functions.

### 5.2 Feature Engineering Classes

**Branch**: `feature/classes-featurize`

**Module**: `src/mlops_2025/features/`

**Architecture**:
- **BaseFeaturesComputer**: Abstract base class defining the feature engineering interface
- **FeaturesComputer**: Concrete implementation for Titanic feature engineering

**Key Benefits**:
- Encapsulates feature engineering logic
- Easy to create alternative feature sets
- Testable and maintainable
- Consistent interface with other components

**Class Structure**:
```python
from mlops_2025.features import FeaturesComputer

# Compute features
features_computer = FeaturesComputer()
df_featurized = features_computer.compute(df)
```

**Features Implemented**:
- Extract Title from Name (Mr, Mrs, Miss, Master, Rare)
- Create Family_size feature (Alone, Small, Large)
- Drop unnecessary columns
- Convert Age to integer

**Files**:
- `base_features_computer.py` - Abstract interface with `compute()` method
- `features_computer.py` - Concrete implementation
- `__init__.py` - Module exports

**Updated Script**: `scripts/featurize.py` now uses the `FeaturesComputer` class.

### 5.3 Model Classes

**Branch**: `feature/classes-models`

**Module**: `src/mlops_2025/models/`

**Architecture**:
- **BaseModel**: Abstract base class defining the model interface
- **LogisticRegressionModel**: Concrete implementation with scikit-learn LogisticRegression

**Key Benefits**:
- Standardized interface for all models
- Easy to swap different model implementations
- Encapsulates preprocessing pipeline
- Consistent train/predict/evaluate API

**Class Structure**:
```python
from mlops_2025.models import LogisticRegressionModel

# Initialize model
model = LogisticRegressionModel(max_iter=1000, random_state=42)

# Train model
trained_pipeline, cv_score = model.train(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Evaluate
accuracy = model.evaluate(X, y)

# Save/Load
model.save('models/model.pkl')
model.load('models/model.pkl')
```

**Features**:
- Built-in preprocessing pipeline (MinMaxScaler, OneHotEncoder, OrdinalEncoder)
- Cross-validation during training
- Support for predict and predict_proba
- Model persistence with pickle

**Files**:
- `base_model.py` - Abstract interface with train/predict/evaluate methods
- `model.py` - LogisticRegressionModel implementation
- `__init__.py` - Module exports

**Updated Scripts**: `scripts/train.py`, `scripts/evaluate.py`, and `scripts/predict.py` now use the `LogisticRegressionModel` class.

### 5.4 RandomForest Model Implementation

**Branch**: `feature/model-rf`

**Module**: `src/mlops_2025/models/random_forest_model.py`

**Implementation**: `RandomForestModel` - Random Forest classifier with preprocessing pipeline

**Usage**:
```python
from mlops_2025.models import RandomForestModel

# Initialize Random Forest model
model = RandomForestModel(n_estimators=100, max_depth=None, random_state=42)

# Train, predict, evaluate (same API as LogisticRegressionModel)
trained_pipeline, cv_score = model.train(X, y)
predictions = model.predict(X)
accuracy = model.evaluate(X, y)
```

**Parameters**:
- `n_estimators`: Number of trees in the forest (default: 100)
- `max_depth`: Maximum depth of trees (default: None, unlimited)
- `random_state`: Random seed for reproducibility

**Features**:
- Same preprocessing pipeline as LogisticRegression
- Ensemble of decision trees
- Better handling of non-linear relationships
- Built-in feature importance

### 5.5 XGBoost Model Implementation

**Branch**: `feature/model-xgb`

**Module**: `src/mlops_2025/models/xgboost_model.py`

**Implementation**: `XGBoostModel` - Gradient boosting classifier with preprocessing pipeline

**Features**:
- Gradient boosting algorithm
- Superior performance on structured data
- Built-in regularization
- Handles missing values automatically
- Feature importance ranking

**Dependencies**: Requires `xgboost>=2.1.3` (added to pyproject.toml)

---

## 6. Model Comparison

All models share the same interface (`BaseModel`) and can be easily swapped:

```python
# Choose any model implementation
from mlops_2025.models import LogisticRegressionModel, RandomForestModel, XGBoostModel

# Logistic Regression
model = LogisticRegressionModel(max_iter=1000, random_state=42)

# Random Forest
model = RandomForestModel(n_estimators=100, random_state=42)

# XGBoost
model = XGBoostModel(n_estimators=100, max_depth=3, random_state=42)

# Same training interface for all
trained_pipeline, cv_score = model.train(X, y)
model.save('models/model.pkl')
```

**Benefits of this architecture**:
-  Easy model comparison
-  Consistent API across all models
-  Swappable implementations
-  Clean separation of concerns
-  Extensible for new models
