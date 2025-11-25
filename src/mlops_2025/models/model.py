import pickle
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        num_features = ['Age', 'Fare']
        cat_features = ['Sex', 'Embarked', 'Title', 'Family_size']
        ord_features = ['Pclass']
        transformers = [
            ('num', MinMaxScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('ord', OrdinalEncoder(), ord_features)
        ]
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        model = SklearnLogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        self.pipeline.fit(X, y)
        
        train_accuracy = self.pipeline.score(X, y)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        return self.pipeline, cv_mean

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet")
        return self.pipeline.score(X, y)

    def save(self, filepath: str) -> None:
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved successfully!")

    def load(self, filepath: str) -> None:
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Model loaded successfully!")
