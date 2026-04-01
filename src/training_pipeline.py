
import os
import sys
import numpy as np
from xgboost import XGBClassifier
from config.constant import *
from logger.logger import logging
from Exception.exception import CustomException
from components.data_transformation import DataTransformer
from components.data_ingestion import DataIngestion
from components.preprocessing import Preprocessing
from components.evaluation import ModelEvaluation
from components.train import ModelTrainer
from components.registry import Registry
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.preprocessing = Preprocessing()
        self.data_transformer = DataTransformer()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        self.model_registry = Registry()

    def run_pipeline(self):
        try:
            logging.info("STEP 1/6 - Data ingestion")
            #Data Ingestion
            train_data = self.data_ingestion.load_data(TRAIN_DATA_PATH)
            test_data = self.data_ingestion.load_data(TEST_DATA_PATH)
            logging.info("Data ingestion completed successfully.")

            # Data Preprocessing
            logging.info("STEP 2/6 - Data preprocessing")
            train_preprocessed = self.preprocessing.transform_data(train_data)
            test_preprocessed = self.preprocessing.transform_data(test_data)


            # Data Transformation
            logging.info("STEP 3/6 - Data transformation")
            x_train_trans, y_train_trans, preprocessor_obj = self.data_transformer.initiate_data_transformation(train_preprocessed)
            x_test_trans, y_test_trans = self.data_transformer.transform_test(test_preprocessed)
            logging.info("Data transformation completed successfully.")

            # Split trained data into train and validation sets
            x_train, y_train, x_val, y_val = self._train_val_split(x_train_trans, y_train_trans)

            # Model Training
            logging.info("STEP 4/6 - Model training")
            results = self.model_trainer.model_training(x_train, y_train, x_val, y_val)
            logging.info("Model training completed successfully.")

            # Model Evaluation
            logging.info("STEP 5/6 - Model evaluation")
            metrics = self.model_evaluation.evaluate_model(results["model"], x_test_trans, y_test_trans)
            logging.info("Model evaluation completed successfully.")

            # Model Registry
            logging.info("STEP 6/6 - Model registry")
            self.model_registry.register_model(results["model"], preprocessor_obj, metrics, results['params'], results['threshold'])

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(f"Error occurred during data ingestion: {e}", sys)

    def _train_val_split(self, X, y, test_size=0.2, random_state=42):
        """Split the data into training and validation sets."""
        train_size = 1-test_size
        x_train = X[:int(train_size*len(X))]
        y_train = y[:int(train_size*len(y))]
        x_val = X[int(train_size*len(X)):]
        y_val = y[int(train_size*len(y)):]
        return x_train, y_train, x_val, y_val
    
if __name__ == "__main__":
    print('Starting the training pipeline...')
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    logging.info("Training pipeline completed successfully.")