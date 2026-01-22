from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, get_chat_template
from datasets import load_dataset, Dataset
from trl import CPOTrainer, CPOConfig
from .base import BaseTrainer
from utils.config import SimPOTrainerConfig
from preprocessing import PreferenceDatasetPreprocessor
import mlflow


class UnslothPreferenceTrainer(BaseTrainer):
    """Trainer for preference learning (DPO) using Unsloth."""

    def __init__(self, config: SimPOTrainerConfig):
        super().__init__(config)

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
                                           chat_template="qwen3-instruct", )

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        dataset = load_dataset(
            "csv",
            data_files=self.config.dataset_path,
            split="train"
        )

        preprocessor = PreferenceDatasetPreprocessor(
            config=self.config, tokenizer=self.tokenizer
        )
        hf_dataset = preprocessor.split(dataset)
        processed_dataset = preprocessor.transform(hf_dataset["train"])
        return processed_dataset

    def setup_trainer(self, dataset):
        """Setup the DPO trainer."""
        training_args = CPOConfig(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,

            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            optim="adamw_8bit",
            weight_decay=self.config.weight_decay,
            lr_scheduler_type="cosine",
            loss_type="simpo",
            cpo_alpha=0.0,

            logging_steps=1,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to="mlflow",
        )

        self.trainer = CPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )

    def train(self):
        """Run the training."""
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("unsloth-nl2sql-SimPO")
        run_name = f"SimPO-{self.config.model_name.split('/')[-1]}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.config.model_dump())

            self.setup_model()
            dataset = self.load_and_preprocess_data()
            self.setup_trainer(dataset)
            self.trainer.train()
            self.save_model()

            mlflow.log_artifacts(
                f"{self.config.output_dir}/lora_model",
                artifact_path="lora_model",
            )

    def save_model(self):
        """Save the trained model."""
        self.model.save_pretrained(f"{self.config.output_dir}/lora_model")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/lora_model")