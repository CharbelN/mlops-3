from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass