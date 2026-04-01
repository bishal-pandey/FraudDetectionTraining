import sys
import numpy as np
import mlflow
import json
import mlflow.sklearn
import numpy as np

from logger.logger import logging
from Exception.exception import CustomException
from config.constant import *

class Registry:
    def __init__(self):
        # os.makedirs('artifacts', exist_ok=True)
        mlflow.set_experiment("Fraud_XGBoost_Experiment")
        self.client = mlflow.tracking.MlflowClient()

    def register_model(self, model, preprocessor_obj, metrics, params, threshold=0.5):
        """This function registers the model in MLFlow model registery"""
        with mlflow.start_run() as run:
            #Log params and metics to MLFlow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            #log model and preprocessor object  to MLFlow model registry
            mlflow.sklearn.log_model(model, 'xgboost_model')
            mlflow.sklearn.log_model(preprocessor_obj, 'preprocessor')
            mlflow.register_model(f'runs:/{run.info.run_id}/xgboost_model', MODEL_NAME)

            # mlflow.log_artifact(FEATURE_NAMES_PATH, artifact_path="features_names")
            mlflow.log_text(json.dumps(FEATURE_NAMES_PATH), "features.json")
            mlflow.log_metric("threshold", threshold)
            logging.info(f"Model registered successfully in MLFlow model registry with name: xgboost_model")
            self.stage_best_model_production()

        return run.info.run_id
    
    def stage_best_model_production(self):
        """This function stages the best model to production in MLFlow model registry"""
        try:
            model_name = MODEL_NAME
            model_versions = self.client.get_latest_versions(model_name)
            best_version = None
            best_f1_score = -1
            for version in model_versions:
                run_id = version.run_id
                metrics = self.client.get_run(run_id).data.metrics
                f1_score = metrics.get('f1_score', None)
                if f1_score is not None and f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_version = version
            
            if best_version:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=best_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
            logging.info(f"Model version {best_version.version} of {model_name} staged to Production successfully.")
        except Exception as e:
            logging.error(f"Error occurred while staging model to Production: {e}")
            raise CustomException(f"Error occurred while staging model to Production: {e}", sys)
    
    
    