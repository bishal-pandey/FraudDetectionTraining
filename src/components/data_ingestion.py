import pandas as pd
import sys
import os
from Exception.exception import CustomException
from logger.logger import logging

class DataIngestion:
    """Class to handle data ingestion from the source."""

    def load_data(self,file_path):
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error occurred while loading data from {file_path}: {e}")
            raise CustomException(f"Error occurred while loading data from {file_path}: {e}", sys)