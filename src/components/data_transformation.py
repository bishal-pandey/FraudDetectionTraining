import joblib
import numpy as np
import pandas as pd
import json
from Exception.exception import CustomException
from logger.logger import logging
from config.constant import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
class DataTransformer:
    """Class to handle data transformation and feature engineering."""
    def __init__(self):
        self.ordinal_encode_cols = ['merchant', 'job','state','city']
        self.onehot_cols = ['category', 'age_group']
        self.num_cols = ['amt', 'distance_km','trans_week_day','trans_month','trans_hour','time_from_last_trans','rolling_avg_amt',	'trans_freq_1hr']
        os.makedirs(TRANSFORM_PIPELINE_DIR, exist_ok=True)
        
    def haversine(self, lat1, lon1, lat2, lon2):
        """"Calculate the distance between user and merchant using longitude and latitude."""
        R = 6371  # Earth radius in km
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c
    
    def transaction_features(self, df):
        """Create new features from the trans_date_trans_time (transaction date and time) column."""
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_week_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['time_from_last_trans']=df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(0)
        df = df.set_index('trans_date_trans_time')
        df['trans_freq_1hr'] = df.groupby('cc_num')['amt'].rolling('1h').count().reset_index(level=0, drop=True) 
        df = df.reset_index()

        return df

    def transaction_amount(self, df):
        """Create new features from the amt (transaction amount) column."""
        df['rolling_avg_amt']=df.groupby('cc_num')['amt'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        return df
    
    def distance_feature(self, df):
        """Create a new feature for the distance between user and merchant."""
        df['distance_km'] = self.haversine(df['lat'],df['long'],df['merch_lat'], df['merch_long'])
        return df
    
    def age_features(self, df):
        """Create new features from the dob (date of birth) column."""
        age = pd.to_datetime(df['trans_date_trans_time'])-pd.to_datetime(df['dob'],format='%Y-%m-%d')
        df['age'] = age.dt.days//365
        df['age_group'] = pd.cut(df['age'], bins=[0,20,40,60,110])
        return df
    
    def encoding_pipeline(self):
        """Define the encoding pipeline for categorical features."""
        
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        onehot_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('ordinal', ordinal_pipeline, self.ordinal_encode_cols),
                ('onehot', onehot_pipeline, self.onehot_cols),
                ('num', num_pipeline, self.num_cols)
            ],
            remainder='passthrough'
        )
        return preprocessor
    
    def _remove_unused_columns(self, df):
        """Remove columns that are not needed for modeling."""
        df = df.drop(columns = ['cc_num','trans_date_trans_time','gender'
                      ,'lat','long','merch_lat','merch_long','age','dob'])
        return df
    
    def save_state(self, preprocessor,  features_names: list):
        """Save the current state of the data transformation."""
        joblib.dump(preprocessor, TRANSFORM_PIPELINE_PATH)

        with open(FEATURE_NAMES_PATH, 'w') as f:
            json.dump(features_names, f)
        

    def initiate_data_transformation(self, df: pd.DataFrame):
        """Initiate the data transformation process."""
        logging.info("Initiating Training data transformation process.")
        y = df['is_fraud']
        x_df = df.drop('is_fraud', axis=1)

        x_df = self.transaction_features(x_df)
        x_df = self.transaction_amount(x_df)
        x_df = self.distance_feature(x_df)
        x_df = self.age_features(x_df)
        x_df = self._remove_unused_columns(x_df)
        preprocessor = self.encoding_pipeline()
        x_train = preprocessor.fit_transform(x_df)
        self.save_state(preprocessor, x_df.columns.tolist())
        logging.info("Data transformation process completed successfully.")
        return x_train, y, preprocessor
    
    def transform_test(self, test_df, lookback_df=None):
        """Called on test data.
        lookback_df = last N days of train — used as read-only context
        for rolling windows, never evaluated."""
        if lookback_df is not None:
            lookback_df['is_lookback'] = True
            test_df['is_lookback'] = False
            combined = pd.concat([lookback_df, test_df], ignore_index=True)
            combined = self.transaction_features(combined)
            combined = self.transaction_amount(combined)
            combined = self.distance_feature(combined)
            combined = self.age_features(combined)
            combined = self._remove_unused_columns(combined)
            df = combined[combined['is_lookback']==False].drop('is_lookback',
                                                                axis=1).reset_index(drop=True)
        else:
            df = self.transaction_features(test_df)
            df = self.transaction_amount(df)
            df = self.distance_feature(df)
            df = self.age_features(df)
            df = self._remove_unused_columns(df)

        y = df['is_fraud']
        x_df = df.drop('is_fraud', axis=1)
        preprocessor = joblib.load(TRANSFORM_PIPELINE_PATH)
        x_df = preprocessor.transform(x_df)
        return x_df, y
    
    