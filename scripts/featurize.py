import argparse
import warnings
from pathlib import Path

import pandas as pd

from mlops_2025.features import FeaturesComputer


def featurize_data(input_path, output_path):
    """Main featurization pipeline."""
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    
    print("Computing features...")
    features_computer = FeaturesComputer()
    df = features_computer.compute(df)
    
    print(f"Featurized data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering for Titanic dataset"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to preprocessed CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for featurized data",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Perform featurization
    df_featurized = featurize_data(args.input, args.output)

    # Save featurized data
    print(f"Saving featurized data to {args.output}...")
    df_featurized.to_csv(args.output, index=False)
    
    print("✓ Feature engineering complete!")
    print(f"Output saved to: {args.output}")
    print(f"Final shape: {df_featurized.shape}")
    
    # Show data info
    if 'Survived' in df_featurized.columns:
        print(f"Training data ready with target column 'Survived'")
    else:
        print(f"Test/inference data ready (no target column)")


if __name__ == "__main__":
    main()
