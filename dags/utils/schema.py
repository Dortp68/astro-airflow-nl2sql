from pydantic import BaseModel, Field
from typing import Optional, List


class BaseTrainerConfig(BaseModel):
    model_name: str
    max_seq_length: int = 2048

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "default"
    run_name: str = "default"
    output_dir: str = "outputs"
    seed: int = 3407


    run_id: str = ""
    experiment_id: str = ""


class SFTTrainerConfig(BaseTrainerConfig):
    dataset_path: str
    dataset_text_field: str = "text"

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"

    packing: bool = False
    use_gradient_checkpointing: str = "unsloth"


class SimPOTrainerConfig(BaseTrainerConfig):
    dataset_path: str
    chosen_column: str = "sql"
    rejected_column: str = "rejected"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    beta: float = 0.1
    use_gradient_checkpointing: str = "unsloth"
