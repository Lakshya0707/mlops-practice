import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from src.logger import logging

mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')
# dagshub.init(repo_owner='Lakshya0707', repo_name='mlops-practice', mlflow=True)

def load_info(path='reports/experiment_info.json') -> dict:
    with open(path) as f:
        return json.load(f)

def register_model(name: str, info: dict):
    uri = f"runs:/{info['run_id']}/{info['model_path']}"
    result = mlflow.register_model(uri, name)
    client = MlflowClient()
    client.set_registered_model_alias(name, 'staging', result.version)
    logging.info('model %s v%s registered, alias: staging', name, result.version)

def main():
    try:
        register_model('iris_model', load_info())
    except Exception as e:
        logging.error('registry failed: %s', e)
        raise

if __name__ == '__main__':
    main()