import argparse
import warnings
from pathlib import Path

import pandas as pd

def extract_title(df):
    """Extract title from Name column."""
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    # Group rare titles
    df['Title'] = df['Title'].replace(
        ['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    return df


def create_family_size(df):
    """Create family size feature from SibSp and Parch."""
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    
    # Categorize family size
    def family_size_category(number):
        if number == 1:
            return "Alone"
        elif 1 < number < 5:
            return "Small"
        else:
            return "Large"
    
    df['Family_size'] = df['Family_size'].apply(family_size_category)
    
    return df


def drop_unnecessary_columns(df):
    """Drop columns that are no longer needed after feature engineering."""
    columns_to_drop = ['Name', 'Parch', 'SibSp', 'Ticket']
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def convert_age_to_int(df):
    """Convert Age column to int64."""
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype('int64')
    return df


def featurize_data(input_path, output_path):
    """Main featurization pipeline."""
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    
    print("Extracting titles from names...")
    df = extract_title(df)
    
    print("Creating family size feature...")
    df = create_family_size(df)
    
    print("Dropping unnecessary columns...")
    df = drop_unnecessary_columns(df)
    
    print("Converting Age to integer...")
    df = convert_age_to_int(df)
    
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
