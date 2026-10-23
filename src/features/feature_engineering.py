import os
import pandas as pd
from src.logger import logging

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['PetalArea'] = df['PetalLengthCm'] * df['PetalWidthCm']
        df['SepalArea'] = df['SepalLengthCm'] * df['SepalWidthCm']
        df['PetalRatio'] = df['PetalLengthCm'] / df['PetalWidthCm']
        df['SepalRatio'] = df['SepalLengthCm'] / df['SepalWidthCm']
        logging.info('features added')
        return df
    except KeyError as e:
        logging.error('missing column: %s', e)
        raise

def save_data(train, test, path='./data/interim'):
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, 'train_feature_engineering.csv'), index=False)
    test.to_csv(os.path.join(path, 'test_feature_engineering.csv'), index=False)
    logging.info('feature engineered data saved to %s', path)

def main():
    try:
        train = add_features(load_data('./data/raw/train.csv'))
        test = add_features(load_data('./data/raw/test.csv'))
        save_data(train, test)
    except Exception as e:
        logging.error('feature engineering failed: %s', e)
        raise

if __name__ == '__main__':
    main()