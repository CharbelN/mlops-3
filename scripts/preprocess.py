"""
Data preprocessing script for Titanic survival prediction.
Handles data loading, cleaning, and basic preprocessing steps.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

# Ignore all warnings
warnings.filterwarnings("ignore")


def load_data(train_path, test_path):
    """Load training and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean_data(train, test):
    """Clean the data by handling missing values and dropping unnecessary columns."""
    # Drop Cabin column due to numerous null values
    train.drop(columns=["Cabin"], inplace=True)
    test.drop(columns=["Cabin"], inplace=True)

    # Fill missing values
    train["Embarked"].fillna("S", inplace=True)
    test["Fare"].fillna(test["Fare"].mean(), inplace=True)

    # Create unified dataframe for easier manipulation
    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    df.corr(numeric_only=True)["Age"].abs()
    # Fill missing Age values using group median
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def split_data(df):
    """Split the unified dataframe back into train and test sets."""
    train = df.loc[:890].copy()
    test = df.loc[891:].copy()

    # Remove Survived column from test set
    if "Survived" in test.columns:
        test.drop(columns=["Survived"], inplace=True)

    # Ensure Survived column is int in train set
    if "Survived" in train.columns:
        train["Survived"] = train["Survived"].astype("int64")

    return train, test


def preprocess_single_file(input_path, output_path):
    """Preprocess a single CSV file (train or test mode)."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")

    # Drop Cabin column due to numerous null values
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    # Fill missing values based on column availability
    if "Embarked" in df.columns:
        df["Embarked"].fillna("S", inplace=True)
    
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].mean(), inplace=True)

    # Fill missing Age values using group median if possible
    if "Age" in df.columns and df["Age"].isnull().any():
        if "Sex" in df.columns and "Pclass" in df.columns:
            df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df["Age"].fillna(df["Age"].median(), inplace=True)

    # Ensure Survived column is int if it exists
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype("int64")

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for preprocessed data",
    )
    parser.add_argument(
        "--test_path", 
        type=str, 
        help="Optional: Path to test CSV file for combined preprocessing"
    )
    parser.add_argument(
        "--output_test",
        type=str,
        help="Optional: Output path for preprocessed test data",
    )

    args = parser.parse_args()

    # Create output directories if they don't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Check if we're doing combined train/test preprocessing or single file
    if args.test_path and args.output_test:
        Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
        
        print("Loading data...")
        train, test = load_data(args.input, args.test_path)
        print(f"Loaded train: {train.shape}, test: {test.shape}")

        print("Cleaning data...")
        df = clean_data(train, test)

        print("Splitting data...")
        train_processed, test_processed = split_data(df)

        print("Saving preprocessed data...")
        train_processed.to_csv(args.output, index=False)
        test_processed.to_csv(args.output_test, index=False)

        print(f"Preprocessed train saved to: {args.output}")
        print(f"Preprocessed test saved to: {args.output_test}")
        print(f"Final train shape: {train_processed.shape}")
        print(f"Final test shape: {test_processed.shape}")
    else:
        # Single file preprocessing
        print("Processing single file...")
        df_processed = preprocess_single_file(args.input, args.output)
        
        print("Saving preprocessed data...")
        df_processed.to_csv(args.output, index=False)
        
        print(f"Preprocessed data saved to: {args.output}")
        print(f"Final shape: {df_processed.shape}")
        print(f"Missing values:\n{df_processed.isnull().sum()}")


if __name__ == "__main__":
    main()
