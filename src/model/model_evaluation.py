import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import logging

mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')
dagshub.init(repo_owner='Lakshya0707', repo_name='mlops-practice', mlflow=True)
mlflow.set_experiment('iris-experiment-v2')

def load_model(path='models/model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision_macro': precision_score(y, preds, average='macro'),
        'recall_macro': recall_score(y, preds, average='macro'),
        'f1_macro': f1_score(y, preds, average='macro'),
    }

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    try:
        model = load_model()
        test = pd.read_csv('./data/interim/test_feature_engineering.csv')
        X_test, y_test = test.drop(columns=['Species']), test['Species']

        with mlflow.start_run() as run:
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_artifact('artifacts/scaler.pkl')

            save_json(metrics, 'reports/metrics.json')
            save_json({'run_id': run.info.run_id, 'model_path': 'model'},
                      'reports/experiment_info.json')
            logging.info('evaluation done, run_id: %s', run.info.run_id)
    except Exception as e:
        logging.error('evaluation failed: %s', e)
        raise

if __name__ == '__main__':
    main()