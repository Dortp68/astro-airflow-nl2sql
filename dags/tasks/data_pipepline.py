from datasets import load_dataset, DatasetDict, ClassLabel
from typing import Optional

def preprocess_instruction_dataset(dataset_path: str,
                                   train: float = 0.8,
                                   test: float = 0.1,
                                   valid: float = 0.1) -> DatasetDict:
    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
        encoding="utf-8",
    )["train"]
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

