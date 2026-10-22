import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) # pandas can not change the data type automatically

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml # for soft chaning 
import logging #log the steps 
from src.logger import logging
# from src.connections import s3_connection

# parameters ko load karna 
def load_params(params_path: str)-> dict:
    ''' Load params from YAML file.'''
    try:
        with open (params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug('parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('file bot find: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logging.eroor('YAML error: %s',e)
        raise
    except Exception as e:
        logging.error('Unexpted error: %s',e)
        raise

# loading the data
def load_data(data_url:str)-> pd.DataFrame:
     ''' load data from a csv file '''
     try:
        df = pd.read_csv(data_url)
        logging.info('data loaded from %s', data_url)
        return df
     except pd.errors.ParserError as e:
         logging.error('failed to parse the csv file: %s',e)
         raise
     except Exception as e:
         logging.error('Unexcepted error occured while loading the data: %s',e)
         raise

def data_preprocessing(df: pd.DataFrame)-> pd.DataFrame:
    ''' pre processing the data'''

    try:
        logging.info('encoding target column.')
        le= LabelEncoder()
        final_df = df.copy()
        final_df['Species'] = le.fit_transform(df['Species'])
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)->None:
    ''' save train and test datasets'''
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok= True)
        train_data.tocsv(os.path.join(raw_data_path,'train.csv'),index= False)
        test_data.tocsv(os.path.join(raw_data_path,'test.csv'),index= False)
        logging.info('train and test data save to &s',raw_data_path)

    except Exception as e:
        logging.error('Unexcepted error occurred while saving the data %s',e)
        raise


def main():
    try:
        params = load_params(params_path='parmas.yaml')
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url='https://github.com/Lakshya0707/mlops-practice/blob/main/notebooks/data.csv')

        final_df = data_preprocessing()
        train_data,test_data = train_test_split(final_df, test_size=test_size,random_state= 42)
        save_data(train_data,test_data,data_path= './data')
    
    except Exception as e:
        logging.error('failed to complete the data ingestion process:%s',e)
        print(f'error:{e}') 

if __name__ == '__main__':
    main()

