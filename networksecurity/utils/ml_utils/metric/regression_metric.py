from networksecurity.entity.artifact_entity import RegressionMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import numpy as np
import sys


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:

        model_r2_score = r2_score(y_true, y_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        regression_metric = RegressionMetricArtifact(
            r2_score=model_r2_score,
            mean_absolute_error=model_mae,
            root_mean_squared_error=model_rmse
        )

        return regression_metric

    except Exception as e:
        raise NetworkSecurityException(e, sys)