from airflow.sdk import dag, task
from datetime import datetime
from dags.dataset import InstructionDatasetGenerator, PreferenceDatasetGenerator
import mlflow
import os
import pandas as pd
import  tempfile


@dag(
    dag_id="generate_datasets_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["nl2sql_dataset", "synthetic", "mlflow"],
)
def generate_datasets_dag():
    @task
    def generate_instruction_dataset():
        generator = InstructionDatasetGenerator()
        generator.generate_dataset(num_samples=10)
        return generator.dataset.dict()

    @task
    def generate_preference_dataset(instruction_dataset):
        from dags.dataset.models import InstructionDataset

        instruction_dataset = InstructionDataset(**instruction_data)
        pref_generator = PreferenceDatasetGenerator(instruction_dataset)
        pref_generator.generate_dataset(num_samples=10)  # Adjust
        return pref_generator.dataset.dict()

    @task
    def log_to_mlflow(instruction_data: dict, preference_data: dict):
        instruction_df = pd.DataFrame(instruction_data)
        preference_df = pd.DataFrame(preference_data)


        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("nl2sql_dataset_generation")

        with mlflow.start_run():
            mlflow.log_param("instruction_num_samples", instruction_df.shape)
            mlflow.log_param("preference_num_samples", preference_df.shape)

            with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    delete=False,
            ) as tmp:
                tmp_path = tmp.name
                instruction_df.to_csv(tmp_path, index=False)
                preference_df.to_csv(tmp_path, index=False)

            try:
                mlflow.log_artifact(tmp_path, artifact_path="datasets")
            finally:
                os.remove(tmp_path)

    instruction_data = generate_instruction_dataset()
    preference_data = generate_preference_dataset(instruction_data)
    log_to_mlflow(instruction_data, preference_data)


generate_datasets_dag()
