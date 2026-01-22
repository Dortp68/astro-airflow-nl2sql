from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from .base import BasePreprocessor
from utils.config import SFTTrainerConfig
from utils.prompts import SYSTEM_PROMPT, DB_SCHEMA


class InstructionDatasetPreprocessor(BasePreprocessor):
    """Preprocessor for instruction datasets (SFT)."""

    def __init__(self, config: SFTTrainerConfig, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__(config)
        self.tokenizer = tokenizer

    def transform(self, dataset: Dataset) -> Dataset:
        """Preprocess instruction dataset by formatting and tokenizing."""

        system_prompt = SYSTEM_PROMPT.format(schema=DB_SCHEMA)

        def format_instruction(example):

            instruction = example.get("instruction", "")
            query = example.get("sql", "")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": query},
            ]

            text = self.tokenizer.apply_chat_template(messages,
                                                      tokenize = False,
                                                      add_generation_prompt = False,
                                                      enable_thinking = False)
            return {"text": text}

        # Format the dataset
        formatted_dataset = dataset.map(
            format_instruction, remove_columns=dataset.column_names
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