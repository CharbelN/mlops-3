import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd

from mlops_2025.models import LogisticRegressionModel


def load_model(model_path):
    """Load trained model from pickle file."""
    print(f"Loading model from {model_path}...")
    model = LogisticRegressionModel()
    model.load(model_path)
    return model


def load_features(input_path):
    """Load featurized data for prediction."""
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_features(df):
    passenger_ids = None
    if "PassengerId" in df.columns:
        passenger_ids = df["PassengerId"].copy()
    
    columns_to_drop = ["PassengerId", "Survived"]
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, passenger_ids


def make_predictions(model, X):
    """Make predictions using the trained model."""
    print("\nMaking predictions...")
    predictions = model.predict(X)
    
    try:
        probabilities = model.predict_proba(X)
        print(f"✓ Generated {len(predictions)} predictions with probabilities")
        return predictions, probabilities
    except AttributeError:
        print(f"✓ Generated {len(predictions)} predictions")
        return predictions, None


def save_predictions(predictions, probabilities, passenger_ids, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = pd.DataFrame()
    
    if passenger_ids is not None:
        results["PassengerId"] = passenger_ids
    
    results["Survived"] = predictions
    
    if probabilities is not None:
        results["Probability_NotSurvived"] = probabilities[:, 0]
        results["Probability_Survived"] = probabilities[:, 1]
    
    print(f"\nSaving predictions to {output_path}...")
    results.to_csv(output_path, index=False)
    print(f"✓ Predictions saved successfully!")
    
    print("\nPrediction Summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted survived: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    print(f"  Predicted not survived: {len(predictions) - sum(predictions)} ({(len(predictions) - sum(predictions))/len(predictions)*100:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained Titanic survival model"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to trained model (pickle file)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to featurized input CSV file (without labels)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for predictions CSV file",
    )

    args = parser.parse_args()

    model = load_model(args.model)
    
    df = load_features(args.input)
    
    X, passenger_ids = prepare_features(df)
    
    predictions, probabilities = make_predictions(model, X)
    
    results = save_predictions(predictions, probabilities, passenger_ids, args.output)
    
    print("Prediction complete!")
    print(f"Output saved to: {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()
