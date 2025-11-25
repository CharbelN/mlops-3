import pandas as pd
from .base_preprocessor import BasePreprocessor


class Preprocessor(BasePreprocessor):

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if "Cabin" in df.columns:
            df.drop(columns=["Cabin"], inplace=True)

        if "Embarked" in df.columns:
            df["Embarked"].fillna("S", inplace=True)
        
        if "Fare" in df.columns:
            df["Fare"].fillna(df["Fare"].mean(), inplace=True)

        if "Age" in df.columns and df["Age"].isnull().any():
            if "Sex" in df.columns and "Pclass" in df.columns:
                df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                df["Age"].fillna(df["Age"].median(), inplace=True)

        if "Survived" in df.columns:
            df["Survived"] = df["Survived"].astype("int64")

        return df


class CombinedPreprocessor(BasePreprocessor):

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if "Cabin" in df.columns:
            df.drop(columns=["Cabin"], inplace=True)

        if "Embarked" in df.columns:
            df["Embarked"].fillna("S", inplace=True)
        
        if "Fare" in df.columns:
            df["Fare"].fillna(df["Fare"].mean(), inplace=True)

        if "Age" in df.columns and df["Age"].isnull().any():
            if "Sex" in df.columns and "Pclass" in df.columns:
                df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                df["Age"].fillna(df["Age"].median(), inplace=True)

        if "Survived" in df.columns:
            df["Survived"] = df["Survived"].astype("int64")

        return df