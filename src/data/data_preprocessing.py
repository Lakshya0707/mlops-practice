import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import logging

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def scale_data(train_df, test_df, target='Species'):
    ''' scaler fill only fit on train '''
    try:
        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_test, y_test = test_df.drop(columns=[target]), test_df[target]

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_train_s[target] = y_train.values
        X_test_s[target] = y_test.values
        logging.info('scaling done')
        return X_train_s, X_test_s, scaler
    except Exception as e:
        logging.error('scaling failed: %s', e)
        raise

def save_data(train_df, test_df, scaler, data_path='./data', artifact_path='artifacts'):
    os.makedirs(os.path.join(data_path, 'processed'), exist_ok=True)
    os.makedirs(artifact_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, 'processed', 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'processed', 'test_processed.csv'), index=False)
    joblib.dump(scaler, os.path.join(artifact_path, 'scaler.pkl'))
    logging.info('processed data and scaler saved')

def main():
    try:
        train = load_data('./data/raw/train.csv')
        test = load_data('./data/raw/test.csv')
        train_s, test_s, scaler = scale_data(train, test)
        save_data(train_s, test_s, scaler)
    except Exception as e:
        logging.error('preprocessing failed: %s', e)
        raise

if __name__ == '__main__':
    main()