import argparse
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mlops_2025.models import LogisticRegressionModel


def load_model(model_path):
    """Load trained model from pickle file."""
    print(f"Loading model from {model_path}...")
    model = LogisticRegressionModel()
    model.load(model_path)
    return model


def load_evaluation_data(input_path):
    """Load featurized evaluation data."""
    print(f"Loading evaluation data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    return df


def prepare_eval_data(df):
    """Prepare features (X) and target (y) for evaluation."""
    if "Survived" not in df.columns:
        raise ValueError("Evaluation data must contain 'Survived' column")
    
    X = df.drop(["Survived", "PassengerId"], axis=1, errors="ignore")
    y = df["Survived"]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    
    cm = confusion_matrix(y, y_pred)
    
    report = classification_report(y, y_pred, target_names=['Not Survived', 'Survived'])
    
    return accuracy, cm, report, y_pred


def print_metrics(accuracy, cm, report):
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\nInterpretation:")
    print(f"  True Negatives:  {cm[0][0]} (correctly predicted not survived)")
    print(f"  False Positives: {cm[0][1]} (incorrectly predicted survived)")
    print(f"  False Negatives: {cm[1][0]} (incorrectly predicted not survived)")
    print(f"  True Positives:  {cm[1][1]} (correctly predicted survived)")
    
    print("\nClassification Report:")
    print(report)


def save_metrics_to_json(accuracy, cm, output_path):
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1])
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving metrics to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Titanic survival prediction model"
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
        help="Path to featurized evaluation CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Output path for metrics JSON file",
    )

    args = parser.parse_args()

    model = load_model(args.model)
    
    df = load_evaluation_data(args.input)
    
    X, y = prepare_eval_data(df)
    
    accuracy, cm, report, y_pred = evaluate_model(model, X, y)
    
    print_metrics(accuracy, cm, report)
    
    if args.output:
        save_metrics_to_json(accuracy, cm, args.output)
    
    print("Evaluation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
