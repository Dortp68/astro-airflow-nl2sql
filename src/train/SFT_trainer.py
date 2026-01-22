from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, get_chat_template
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from .base import BaseTrainer
from .callbacks import StopTrainingCallback
from utils.config import SFTTrainerConfig
from preprocessing import InstructionDatasetPreprocessor
import mlflow


class UnslothInstructionTrainer(BaseTrainer):
    """Trainer for supervised fine-tuning using Unsloth."""

    def __init__(self, config: SFTTrainerConfig):
        super().__init__(config)
        self.train_on_completions=True

    def setup_model(self):
        """Load and configure the model with LoRA."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            full_finetuning=self.config.full_finetuning,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.seed,
        )

        self.tokenizer = get_chat_template(self.tokenizer,
                                           chat_template = "qwen3-instruct",)

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        dataset = load_dataset(
            "csv",
            data_files=self.config.dataset_path,
            split="train"
        )

        preprocessor = InstructionDatasetPreprocessor(
            config=self.config, tokenizer=self.tokenizer
        )
        hf_dataset = preprocessor.split(dataset)
        processed_dataset = preprocessor.transform(hf_dataset["train"])
        return processed_dataset

    def setup_trainer(self, dataset: Dataset, train_on_completions: bool = True):
        """Set up the SFT trainer."""

        training_args = SFTConfig(
            dataset_text_field=self.config.dataset_text_field,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_length=self.config.max_seq_length,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,

            logging_strategy="steps",
            logging_steps=1,

            save_strategy="epoch",
            output_dir="./SFTcheckpoints",

            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            packing=self.config.packing,
            report_to="mlflow",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[StopTrainingCallback(self.stop_event)],
        )


        if self.train_on_completions:
            self.trainer = train_on_responses_only(
                self.trainer,
                instruction_part="<|im_start|>system\n",
                response_part="<|im_start|>assistant\n",
            )

    def train(self, train_on_completions: bool = True):
        """Run the training."""
        self.train_on_completions = train_on_completions
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=self.config.run_name) as run:
            self.config.experiment_id = run.info.experiment_id
            self.config.run_id = run.info.run_id
            mlflow.log_params(self.config.model_dump())

            self.setup_model()
            dataset = self.load_and_preprocess_data()
            self.setup_trainer(dataset)

            try:
                self.trainer.train()
            finally:
                self.save_model()

                mlflow.log_artifacts(
                    f"{self.config.output_dir}/lora_model",
                    artifact_path="lora_model",
                )

    def save_model(self):
        """Save the trained model."""
        self.model.save_pretrained(f"{self.config.output_dir}/lora_model")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/lora_model")