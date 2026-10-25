# promote model

import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = 'iris_model'

def promote_model():
    if not os.getenv('MLFLOW_TRACKING_USERNAME') or not os.getenv('MLFLOW_TRACKING_PASSWORD'):
        raise EnvironmentError('MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD set nahi hai')

    mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')
    client = MlflowClient()

    # staging alias wala latest version
    staging = client.get_model_version_by_alias(MODEL_NAME, 'staging')

    # production alias is version pe laga do (purana version apne aap hat jata hai)
    client.set_registered_model_alias(MODEL_NAME, 'production', staging.version)
    print(f'{MODEL_NAME} v{staging.version} promoted to production')

if __name__ == '__main__':
    promote_model()