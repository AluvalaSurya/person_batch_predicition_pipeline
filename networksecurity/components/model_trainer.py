import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.regression_metric import get_regression_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

import mlflow
# from urllib.parse import urlparse
    
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)
mlflow.set_experiment(
    "student_arrival_regression"
)

class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, regression_metric):

        try:
            # tracking_url_type_store = urlparse(
            #     mlflow.get_tracking_uri()
            # ).scheme

            with mlflow.start_run():

                mlflow.log_metric("r2_score",regression_metric.r2_score)
                mlflow.log_metric("mean_absolute_error",regression_metric.mean_absolute_error)
                mlflow.log_metric("root_mean_squared_error",regression_metric.root_mean_squared_error)

                mlflow.sklearn.log_model(best_model,"model")

                # if tracking_url_type_store != "file":
                #     mlflow.sklearn.log_model(best_model,"model",
                #         registered_model_name=type(best_model).__name__
                #     )

                # else:
                #     mlflow.sklearn.log_model(best_model,"model")

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def train_model(self, X_train, y_train, X_test, y_test):

        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {

                "Linear Regression": {},

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse']
                },

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128]
                },

                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128]
                },

                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            ## Best model score
            best_model_score = max(sorted(model_report.values()))

            ## Best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            ## Best model
            best_model = models[best_model_name]

            ## Train best model
            best_model.fit(X_train, y_train)

            ## Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            ## Train metrics
            regression_train_metric = get_regression_score(
                y_true=y_train,
                y_pred=y_train_pred
            )

            ## Test metrics
            regression_test_metric = get_regression_score(
                y_true=y_test,
                y_pred=y_test_pred
            )

            ## MLflow tracking
            self.track_mlflow(best_model,regression_test_metric)

            ## Load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            ## Create model directory
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(model_dir_path, exist_ok=True)

            ## Final model object
            network_model = NetworkModel(preprocessor=preprocessor,model=best_model)

            ## Save trained model
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            ## Save final model separately
            save_object(
                "final_model/model.pkl",
                best_model
            )

            ## Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=
                self.model_trainer_config.trained_model_file_path,

                train_metric_artifact=regression_train_metric,

                test_metric_artifact=regression_test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## Load arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            ## Split features and target
            X_train, y_train, X_test, y_test = (

                train_arr[:, :-1],
                train_arr[:, -1],

                test_arr[:, :-1],
                test_arr[:, -1]
            )

            ## Train model
            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)