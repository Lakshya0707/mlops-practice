import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO

# configure logging
# logging.basicCongfig(level = logging.info)
# logger = logging.getlogger(__name__)


class s3_operations:
    def __init__(self,bucket_name, aws_access_key, aws_secret_key ,region_name = 'us-east-1'):
        ''' initialize the s3_operations class with AWS Credentials and s3 bucket details.'''

        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id = aws_access_key,
            aws_secret_access_key = aws_secret_key,
            region_name = region_name
        )

        logging.info('DataIngestion from s3 bucket initialized.')

        def fetch_file_from_s3( self,file_key):
            ''' fetches csv file from the s3 bucket and returns it as a pandas Data fram.
            :params file_key : s3 file path (eg., 'data/data.csv)
            :returns :pandas Data frame'''

            try:
                logging.ino(f"fetching file '{file_key}' from s3 bucket '{self.bucket_name}'...")
                obj = self.s3_client.get_object(Bucket = self.bucket_name , key = file_key)
                df = pd.read_csv(StringIO(obj['Body'.read().decode('utf-8')]))
                logging.info(f"successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
                return df
            except Exception as e:
                logging.exceptin(f"failed to fetch '{file_key} from s3: {e}")
                return None
