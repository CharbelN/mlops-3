import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd

from mlops_2025.models import LogisticRegressionModel


def load_featurized_data(input_path):
    """Load featurized data."""
    print(f"Loading featurized data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_train_data(df):
    """Prepare features (X) and target (y) for training."""
    if "Survived" not in df.columns:
        raise ValueError("Training data must contain 'Survived' column")
    
    X = df.drop(["Survived", "PassengerId"], axis=1, errors="ignore")
    y = df["Survived"]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train Titanic survival prediction model"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to featurized training CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for trained model (pickle file)",
    )

    args = parser.parse_args()

    # Load data
    df = load_featurized_data(args.input)
    X, y = prepare_train_data(df)
    
    # Create and train model
    print("\nInitializing Logistic Regression model...")
    model = LogisticRegressionModel(max_iter=1000, random_state=42)
    
    trained_pipeline, cv_accuracy = model.train(X, y)
    
    # Save model
    model.save(args.output)
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
