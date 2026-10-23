import os
import pickle
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_params(path='params.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)['model_building']

def train_model(X, y,params):
    try:
        model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                        max_depth= params['max_depth'],
                                        random_state= params['random_state'])
        model.fit(X, y)
        logging.info('model trained')
        return model
    except Exception as e:
        logging.error('training failed: %s', e)
        raise

def save_model(model, path='models/model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logging.info('model saved to %s', path)

def main():
    try:
        params = load_params()
        train = load_data('./data/interim/train_feature_engineering.csv')
        X_train, y_train = train.drop(columns=['Species']), train['Species']
        model = train_model(X_train, y_train,params)
        save_model(model)
    except Exception as e:
        logging.error('model building failed: %s', e)
        raise

if __name__ == '__main__':
    main()