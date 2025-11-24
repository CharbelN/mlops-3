import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

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


def build_preprocessing_pipeline():
    num_features = ['Age', 'Fare']
    cat_features = ['Sex', 'Embarked', 'Title', 'Family_size']
    ord_features = ['Pclass']
    
    transformers = []
    
    # Numeric features: MinMaxScaler
    transformers.append(('num', MinMaxScaler(), num_features))
    
    # Categorical features: OneHotEncoder
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_features))
    
    # Ordinal features: OrdinalEncoder
    transformers.append(('ord', OrdinalEncoder(), ord_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    return preprocessor


def build_model_pipeline():
    preprocessor = build_preprocessing_pipeline()
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline


def train_model(X, y, pipeline):
    print("\nTraining model...")
    
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nTraining on full dataset...")
    pipeline.fit(X, y)
    
    train_accuracy = pipeline.score(X, y)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    return pipeline, cv_scores.mean()


def save_model(pipeline, output_path, cv_accuracy):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved successfully!")
    print(f"Cross-validation accuracy: {cv_accuracy:.4f}")


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

    df = load_featurized_data(args.input)
    
    X, y = prepare_train_data(df)
    
    print("\nBuilding model pipeline...")
    pipeline = build_model_pipeline()
    
    trained_pipeline, cv_accuracy = train_model(X, y, pipeline)
    
    save_model(trained_pipeline, args.output, cv_accuracy)
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Model saved to: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
