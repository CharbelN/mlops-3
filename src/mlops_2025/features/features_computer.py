import pandas as pd
from .base_features_computer import BaseFeaturesComputer


class FeaturesComputer(BaseFeaturesComputer):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._extract_title(df)
        df = self._create_family_size(df)
        df = self._drop_unnecessary_columns(df)
        df = self._convert_age_to_int(df)
        
        return df

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
            
            df['Title'] = df['Title'].replace(
                ['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
            )
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        return df

    def _create_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['Family_size'] = df['SibSp'] + df['Parch'] + 1
            
            def family_size_category(number):
                if number == 1:
                    return "Alone"
                elif 1 < number < 5:
                    return "Small"
                else:
                    return "Large"
            
            df['Family_size'] = df['Family_size'].apply(family_size_category)
        
        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = ['Name', 'Parch', 'SibSp', 'Ticket']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        return df

    def _convert_age_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in df.columns:
            df['Age'] = df['Age'].astype('int64')
        return df
