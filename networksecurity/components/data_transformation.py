# import sys
# import os
# import numpy as np
# import pandas as pd
# from sklearn.impute import KNNImputer
# from sklearn.pipeline import Pipeline

# from networksecurity.constant.training_pipeline import TARGET_COLUMN
# from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

# from networksecurity.entity.artifact_entity import (
#     DataTransformationArtifact,
#     DataValidationArtifact
# )

# from networksecurity.entity.config_entity import DataTransformationConfig
# from networksecurity.exception.exception import NetworkSecurityException 
# from networksecurity.logger.logger import logging
# from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object

# class DataTransformation:
#     def __init__(self,data_validation_artifact:DataValidationArtifact,
#                  data_transformation_config:DataTransformationConfig):
#         try:
#             self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
#             self.data_transformation_config:DataTransformationConfig=data_transformation_config
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)
        
#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise NetworkSecurityException(e, sys)
        
#     def get_data_transformer_object(cls)->Pipeline:
#         """
#         It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
#         and returns a Pipeline object with the KNNImputer object as the first step.

#         Args:
#           cls: DataTransformation

#         Returns:
#           A Pipeline object
#         """
#         logging.info(
#             "Entered get_data_trnasformer_object method of Trnasformation class"
#         )
#         try:
#            imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
#            logging.info(
#                 f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
#             )
#            processor:Pipeline=Pipeline([("imputer",imputer)])
#            return processor
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)

        
#     def initiate_data_transformation(self)->DataTransformationArtifact:
#         logging.info("Entered initiate_data_transformation method of DataTransformation class")
#         try:
#             logging.info("Starting data transformation")
#             train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
#             test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

#             ## training dataframe
#             input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]
#             target_feature_train_df = target_feature_train_df.replace(-1, 0)

#             #testing dataframe
#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]
#             target_feature_test_df = target_feature_test_df.replace(-1, 0)

#             preprocessor=self.get_data_transformer_object()

#             preprocessor_object=preprocessor.fit(input_feature_train_df)
#             transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
#             transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
             

#             train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
#             test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

#             #save numpy array data
#             save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
#             save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
#             save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

#             save_object( "final_model/preprocessor.pkl", preprocessor_object,)


#             #preparing artifacts

#             data_transformation_artifact=DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#             )
#             return data_transformation_artifact


            
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)


import sys
import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constant.training_pipeline import TARGET_COLUMN,DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,data_transformation_config: DataTransformationConfig):

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates preprocessing pipeline
        """
        try:
            logging.info("Entered get_data_transformer_object method")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor = Pipeline(steps=[("imputer", imputer)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Read raw validated data
            df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)

            # Timestamp conversion
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Minutes column
            df["minutes"] = (df["timestamp"].dt.hour * 60+ df["timestamp"].dt.minute)

            # Day of week feature
            df["day_of_week"] = (df["timestamp"].dt.dayofweek)

            # Sort values
            df = df.sort_values(["person", "device", "timestamp"])
            df.head()

            # Previous day arrival
            df["previous_day_arrival"] = df.groupby(["person", "device"])["minutes"].shift(1)
            
            # Remove extreme outliers
            cutoff = df["minutes"].quantile(0.95)
            df = df[df["minutes"] <= cutoff].reset_index(drop=True)

            # Rolling mean
            df["rolling_mean"] = df.groupby(["person", "device"])["minutes"].transform(
                lambda x: x.shift(1).rolling(5).mean()
            )
            
            # Rolling std
            df["rolling_std"] = df.groupby(["person", "device"])["minutes"].transform(
                lambda x: x.shift(1).rolling(5).std()
            )
            
            # Drop null rows
            df = df.dropna().reset_index(drop=True)

            # Student mapping
            person_map = (df[["person"]].drop_duplicates().reset_index(drop=True))
            person_map["person_id"] = (person_map.index)

            # Tracker mapping
            device_map = (df[["device"]].drop_duplicates().reset_index(drop=True))
            device_map["device_id"] = (device_map.index)

            # Merge mappings
            df = df.merge(person_map,on="person",how="left")
            df = df.merge(device_map,on="device",how="left")

            # Create mapping directory
            person_mapping_path = self.data_transformation_config.student_mapping_file_path
            device_mapping_path = self.data_transformation_config.tracker_mapping_file_path
  
            os.makedirs(os.path.dirname(person_mapping_path),exist_ok=True)
            
            # Save mapping files
            person_map.to_csv(person_mapping_path,index=False)
            logging.info(f"Student mapping file saved at: {person_mapping_path}")
            device_map.to_csv(device_mapping_path,index=False)
            logging.info(f"Student mapping file saved at: {device_mapping_path}")
                         
            # Convert datatype
            df["previous_day_arrival"] = (df["previous_day_arrival"].astype(int))

            # Student-wise sequential split
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()

            for person_id, group in df.groupby("person_id"):
                group = group.sort_values("timestamp")

                # train_size = int(len(group) * 0.8)
                train_size = int(len(group) * self.data_transformation_config.train_test_split_ratio)
                train_group = group.iloc[:train_size]
                test_group = group.iloc[train_size:]

                train_df = pd.concat([train_df, train_group])
                test_df = pd.concat([test_df, test_group])

            # Drop unnecessary columns
            drop_cols = ["person", "device", "date", "timestamp"]
            train_df = train_df.drop(columns=drop_cols)
            test_df = test_df.drop(columns=drop_cols)
            
            # Save sample batch file
            test_batch_df = test_df.drop(columns=[TARGET_COLUMN])
            os.makedirs(os.path.dirname(
                self.data_transformation_config.batch_prediction_file_path),
                exist_ok=True
            )

            test_batch_df.to_csv(
                self.data_transformation_config.batch_prediction_file_path,
                index=False
            )

            logging.info(
                f"Batch prediction file saved at: "
                f"{self.data_transformation_config.batch_prediction_file_path}"
            )


            # Split input and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Preprocessor
            preprocessor = (self.get_data_transformer_object())
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            

            # Create arrays
            train_arr = np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

            # Save arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)

            # Save preprocessor
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)
            save_object("final_model/preprocessor.pkl",preprocessor_object)

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

            logging.info("Data transformation completed")

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)