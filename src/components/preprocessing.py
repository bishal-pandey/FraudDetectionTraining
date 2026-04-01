import sys
import pandas as pd
from Exception.exception import CustomException
from logger.logger import logging

class Preprocessing:
    """Class to handle data preprocessing and feature engineering."""    
    def _drop_duplicates(self, df):
        """Drop duplicate rows from the dataframe."""
        initial_shape = df.shape
        df = df.drop_duplicates()
        logging.info(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
        return df
    
    def _fix_dtypes(self, df):
        """Enforce correct data types for the dataframe"""
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
        return df
    
    def _remove_unnecessary_columns(self, df):
        """Remove unnecessary columns from the dataframe."""
        remove_col = ['Unnamed: 0','first','last','trans_num','unix_time','street','zip','city_pop']
        df = df.drop(remove_col, axis=1, errors='ignore')
        return df
    
    def _sort_chronologically(self, df):
        """
        Must sort by time before any rolling/velocity features are computed.
        Feature engineering depends on this ordering being correct.
        """
        df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
        return df
    
    def transform_data(self, df):
        """Run the full preprocessing pipeline on the dataframe."""
        try:
            logging.info("Starting data preprocessing...")
            df = self._drop_duplicates(df)
            df = self._fix_dtypes(df)
            df = self._remove_unnecessary_columns(df)
            df = self._sort_chronologically(df)
            logging.info("Data preprocessing completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            raise CustomException(f"Error occurred during data preprocessing: {e}", sys.exc_info())