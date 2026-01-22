from airflow.sdk import dag, task
from datetime import datetime
from dags.dataset import InstructionDatasetGenerator
import mlflow
import os
import pandas as pd
import  tempfile


@dag(
    dag_id="generate_instruction_dataset_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["nl2sql_dataset", "instruction", "synthetic"],
)
def generate_instruction_dataset_dag():
    @task
    def generate_instruction_dataset(num_samples: int = 100):
        generator = InstructionDatasetGenerator()
        generator.generate_dataset(num_samples=num_samples)
        return generator.dataset.dict()

    @task
    def save_dataset_to_file(dataset_dict):
        instruction_df = pd.DataFrame(dataset_dict)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("nl2sql_preference_dataset_generation")

        with mlflow.start_run():
            mlflow.log_param("instruction_num_samples", instruction_df.shape)

            with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    delete=False,
            ) as tmp:
                tmp_path = tmp.name
                instruction_df.to_csv(tmp_path, index=False)

            try:
                mlflow.log_artifact(tmp_path, artifact_path="datasets")
            finally:
                os.remove(tmp_path)

    dataset = generate_instruction_dataset()
    save_dataset_to_file(dataset)


generate_instruction_dataset_dag()
