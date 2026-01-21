from airflow.sdk import dag, task
from datetime import datetime
from dags.utils.schema import SFTTrainerConfig
import requests

@dag(
    dag_id="healthcheck_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["preference", "mlflow", "from_instruction"],
)
def healthcheck_dag():

    @task
    def check_health():
        url = "http://api:8000/monitoring/health"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        print("OK:", resp.text)

    @task
    def start_training():
        url = "http://api:8000/train/sft"
        config = SFTTrainerConfig(model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
                                  max_seq_length = 4096,
                                  output_dir="nl2sql_sft_lora",
                                  dataset_path="./data/instruction/nl2sql_instruction_dataset.csv",
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
    # check_health()

healthcheck_dag()