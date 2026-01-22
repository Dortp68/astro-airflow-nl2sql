from airflow.sdk import dag, task
from datetime import datetime
from dags.utils.schema import SimPOTrainerConfig
import requests

@dag(
    dag_id="simpo_training_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["SimPO", "training"],
)
def simpo_training_dag():
    @task
    def start_training():
        url = "http://api:8000/train/simpo"
        config = SimPOTrainerConfig(model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
                                  output_dir="nl2sql_simpo_lora",
                                  dataset_path="./data/preference/nl2sql_preference_dataset.csv",
                                  packing=False,
                                  use_gradient_checkpointing="unsloth",
                                  experiment_name="nl2sql_sft_lora",
                                  run_name="Qwen3-4B-Instruct")
        resp = requests.post(url,
                             json=config.dict(),
                             timeout=30,)

        resp.raise_for_status()
        print(resp.json())

    start_training()

simpo_training_dag()