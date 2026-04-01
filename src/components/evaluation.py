import sys
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
from config.constant import *
from logger.logger import logging   
from Exception.exception import CustomException


class ModelEvaluation:
    def evaluate_model(self, model, x_test, y_test):
        """This function evaluates the model performance on the test set and logs the metrics."""
        try:
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            logging.info(f"Model Performance Metrics:")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Recall: {recall}")
            logging.info(f"Precision: {pr_auc}")

            return {
                "f1_score": f1,
                "recall": recall,
                "precision": pr_auc
            }

        except Exception as e:
            logging.error(f"Error occurred during model evaluation: {e}")
            raise CustomException(f"Error occurred during model evaluation: {e}", sys)
