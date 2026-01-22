from typing import Dict, List, Any
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from .base import BasePreprocessor
from utils.config import BaseTrainerConfig


class PreferenceDatasetPreprocessor(BasePreprocessor):
    """Preprocessor for preference datasets (SimPO)."""

    def __init__(self, config: BaseTrainerConfig, tokenizer: AutoTokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.chosen_column = config.chosen_column
        self.rejected_column = config.rejected_column

    def transform(self, dataset: Dataset) -> Dataset:
        """Preprocess instruction dataset by formatting and tokenizing."""
        def format_preference(example):

            instruction = example.get("instruction", "")
            chosen = example.get(self.chosen_column, "")
            rejected = example.get(self.rejected_column, "")

            chosen_messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": chosen}
            ]
            rejected_messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": rejected}
            ]

            chosen_text = self.tokenizer.apply_chat_template(chosen_messages,
                                                      tokenize=False,
                                                      add_generation_prompt=False,
                                                      enable_thinking=False)

            rejected_text = self.tokenizer.apply_chat_template(rejected_messages,
                                                             tokenize=False,
                                                             add_generation_prompt=False,
                                                             enable_thinking=False)

            return {"chosen": chosen_text,
                    "rejected": rejected_text}

        formatted_dataset = dataset.map(
            format_preference, remove_columns=dataset.column_names
        )

        return formatted_dataset

    def split(self, dataset: Dataset,
              train: float = 0.8,
              test: float = 0.1,
              valid: float = 0.1) -> DatasetDict:
        dataset = dataset.shuffle()
        labels = sorted(dataset.unique("complexity_level"))
        dataset = dataset.cast_column(
            "complexity_level",
            ClassLabel(names=[str(x) for x in labels])
        )

        train_temp = dataset.train_test_split(
            test_size=(1.0 - train),
            stratify_by_column="complexity_level",
        )
        train_ds = train_temp["train"]
        temp_ds = train_temp["test"]
        valid_test = temp_ds.train_test_split(
            test_size=test / (valid + test),
            stratify_by_column="complexity_level",
        )
        valid_ds = valid_test["train"]
        test_ds = valid_test["test"]

        dataset_dict = DatasetDict({
            "train": train_ds,
            "validation": valid_ds,
            "test": test_ds,
        })

        return dataset_dict
