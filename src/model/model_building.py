import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_model(X, y):
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
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
        train = load_data('./data/interim/train_feature_engineering.csv')
        X_train, y_train = train.drop(columns=['Species']), train['Species']
        model = train_model(X_train, y_train)
        save_model(model)
    except Exception as e:
        logging.error('model building failed: %s', e)
        raise

if __name__ == '__main__':
    main()