import sys
import json
import numpy as np 
from xgboost import XGBClassifier
from config.constant import *
from logger.logger import logging
from Exception.exception import CustomException


class ModelTrainer:
    def __init__(self):
        self.params = {
            "n_estimators": MODEL_N_ESTIMATORS,
            "max_depth": MODEL_MAX_DEPTH,
            "min_child_weight": MODEL_MIN_CHILD_WEIGHT,
            "gamma": MODEL_GAMMA,
            "subsample": MODEL_SUBSAMPLE,
            "colsample_bytree": MODEL_COLSAMPLE_BYTREE,
            "learning_rate": MODEL_LEARNING_RATE,
            "reg_alpha": MODEL_REG_ALPHA,
            "scale_pos_weight": MODEL_SCALE_POS_WEIGHT,
            "objective": MODEL_OBJECTIVE,
            "eval_metric": MODEL_EVAL_METRIC
        }
    def model_training(self, x_train, y_train, x_test, y_test):
        '''This function train XgBoost classifier with specrified parameters'''
        try:
            logging.info("Training XgBoost classifier with specified parameters")

            # Train the model
            self.model = XGBClassifier(**self.params)

            logging.info("Model training going on...")
            self.model.fit(x_train, 
                           y_train,
                            eval_set=[(x_test, y_test)],
                            verbose=False
                           )
            logging.info("Model training done.")
            # best_iteration = self.model.best_iteration
            # best_score     = self.model.best_score
            # logging.info(f"Best iteration : {best_iteration}  "
            #             f"val score = {best_score:.4f}")

            with open(FEATURE_NAMES_PATH, "r") as f:
                features_names = json.load(f)

            self._log_feature_importance(self.model, feature_names=features_names, top_n=10)
            return {
                "model": self.model,
                'params': self.params,
                'threshold': 0.5
            }

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(f"Error occurred during model training: {e}", sys)

    def _log_feature_importance(self, model, feature_names, top_n=10):
        '''This function logs the feature importance of the trained model'''
        importances = model.feature_importances_
        top = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        logging.info("  Top features by importance:")
        for name, score in top:
            logging.info(f"{name} {score:.4f}")