from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datasets import Dataset, DatasetDict
from utils.config import BaseTrainerConfig


class BasePreprocessor(ABC):
    """Base class for dataset preprocessing."""

    def __init__(self, config: BaseTrainerConfig):
        self.config = config

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset.

        Args:
            dataset: Raw dataset to preprocess

        Returns:
            Preprocessed dataset
        """
        pass

    @abstractmethod
    def split(self, dataset: Dataset,
              train: float,
              test: float,
              valid: float) -> DatasetDict:
        """Split the dataset.
        
        Args:
            dataset: Raw dataset to split
            train: Percentage of training data
            test: Percentage of testing data
            valid: Percentage of testing data

        Returns:
            Dictionary of splitted dataset
        """
        pass
