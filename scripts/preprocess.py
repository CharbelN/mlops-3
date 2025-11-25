import argparse
import warnings
from pathlib import Path

import pandas as pd

from mlops_2025.preprocessing import Preprocessor, CombinedPreprocessor

def load_data(train_path, test_path):
    """Load training and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


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

        print("Preprocessing data...")
        # Combine train and test for consistent preprocessing
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        
        # Use CombinedPreprocessor for combined preprocessing
        preprocessor = CombinedPreprocessor()
        df_processed = preprocessor.process(df)

        print("Splitting data...")
        train_processed, test_processed = split_data(df_processed)

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
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Loaded data shape: {df.shape}")
        
        # Use standard Preprocessor for single file
        preprocessor = Preprocessor()
        df_processed = preprocessor.process(df)
        
        print("Saving preprocessed data...")
        df_processed.to_csv(args.output, index=False)
        
        print(f"Preprocessed data saved to: {args.output}")
        print(f"Final shape: {df_processed.shape}")
        print(f"Missing values:\n{df_processed.isnull().sum()}")


if __name__ == "__main__":
    main()
