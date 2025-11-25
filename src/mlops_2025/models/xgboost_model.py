"""XGBoost model implementation."""

import pickle
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model with preprocessing pipeline."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """Initialize the XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Step size shrinkage used to prevent overfitting
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the preprocessing and model pipeline."""
        # Define feature groups
        num_features = ['Age', 'Fare']
        cat_features = ['Sex', 'Embarked', 'Title', 'Family_size']
        ord_features = ['Pclass']
        
        # Create transformers
        transformers = [
            ('num', MinMaxScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('ord', OrdinalEncoder(), ord_features)
        ]
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Create model
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        """Train the model on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (trained_pipeline, cross_validation_score)
        """
        print("\nTraining XGBoost model...")
        
        # Perform cross-validation
        print("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Train on full dataset
        print("\nTraining on full dataset...")
        self.pipeline.fit(X, y)
        
        train_accuracy = self.pipeline.score(X, y)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        return self.pipeline, cv_mean

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on the provided data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities on the provided data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate the model on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Accuracy score
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.score(X, y)

    def save(self, filepath: str) -> None:
        """Save the model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved successfully!")

    def load(self, filepath: str) -> None:
        """Load the model from a file.
        
        Args:
            filepath: Path from where to load the model
        """
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Model loaded successfully!")
