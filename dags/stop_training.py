from airflow.sdk import dag, task
from datetime import datetime
import requests

@dag(
    dag_id="stop_training_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["preference", "mlflow", "from_instruction"],
)
def stop_training_dag():
    @task
    def stop_training():
        url = "http://api:8000/train/stop"
        resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        print(resp.json())

    stop_training()

stop_training_dag()