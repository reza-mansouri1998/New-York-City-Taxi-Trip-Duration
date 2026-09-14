import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration for saving the preprocessor object."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the ColumnTransformer object based on the notebook's pipeline.
        """
        try:
            numerical_cols = [
                'passenger_count', 
                'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude',
                'pickup_hour', 'pickup_dayofweek', 'pickup_month',
                'haversine_dist_km' 
            ]
            
            categorical_cols = ['vendor_id', 'store_and_fwd_flag']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ],
                remainder='drop'
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculates the physical distance between two geographical points."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def _feature_engineering(self, df):
        """Applies datetime extraction and distance calculation to the dataframe."""
        try:
            # Parse datetime if it's read as string
            if not pd.api.types.is_datetime64_any_dtype(df['pickup_datetime']):
                df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

            # Extract temporal features
            df['pickup_hour'] = df['pickup_datetime'].dt.hour
            df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
            df['pickup_month'] = df['pickup_datetime'].dt.month

            # Calculate Haversine distance
            df['haversine_dist_km'] = self.haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )

            # Drop unused columns
            columns_to_drop = ['id', 'pickup_datetime', 'dropoff_datetime']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train/test data, applies feature engineering, transforms features, 
        and saves the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            # Domain-Knowledge Outlier Filtering (Applied only to training data to preserve test representation)
            logging.info("Filtering outliers in training dataset")
            valid_idx = (train_df['trip_duration'] >= 60) & (train_df['trip_duration'] <= 7200)
            train_df = train_df[valid_idx].copy()

            logging.info("Applying feature engineering to train and test datasets")
            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "trip_duration"

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Apply Log1p transformation to target variables to optimize for RMSLE
            target_feature_train_arr = np.log1p(target_feature_train_df.values)
            target_feature_test_arr = np.log1p(target_feature_test_df.values)

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit and transform training data, only transform testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features and log-transformed target into final arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)