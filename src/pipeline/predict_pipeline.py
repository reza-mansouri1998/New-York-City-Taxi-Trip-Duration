import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Point to the serialized model and column transformer
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculates the physical distance between two geographical points."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def predict(self, features):
        """Engineers features, applies scaling, and returns the actual predicted seconds."""
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Recreate temporal features from the HTML form's datetime string
            features['pickup_datetime'] = pd.to_datetime(features['pickup_datetime'])
            features['pickup_hour'] = features['pickup_datetime'].dt.hour
            features['pickup_dayofweek'] = features['pickup_datetime'].dt.dayofweek
            features['pickup_month'] = features['pickup_datetime'].dt.month

            # Recreate spatial distance
            features['haversine_dist_km'] = self.haversine_distance(
                features['pickup_latitude'], features['pickup_longitude'],
                features['dropoff_latitude'], features['dropoff_longitude']
            )

            # Drop the raw datetime column so it matches the preprocessor's expected input
            features = features.drop(columns=['pickup_datetime'])

            # Scale data and predict
            data_scaled = preprocessor.transform(features)
            preds_log = model.predict(data_scaled)
            
            # Inverse the log1p transformation applied during the training phase
            preds_actual = np.expm1(preds_log)
            
            return preds_actual
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Acts as a bridge between the Flask HTML form and the PredictPipeline.
    Ensures data types are strictly enforced before conversion to a DataFrame.
    """
    def __init__(self,
                 vendor_id: int,
                 passenger_count: int,
                 pickup_longitude: float,
                 pickup_latitude: float,
                 dropoff_longitude: float,
                 dropoff_latitude: float,
                 store_and_fwd_flag: str,
                 pickup_datetime: str):
        
        self.vendor_id = vendor_id
        self.passenger_count = passenger_count
        self.pickup_longitude = pickup_longitude
        self.pickup_latitude = pickup_latitude
        self.dropoff_longitude = dropoff_longitude
        self.dropoff_latitude = dropoff_latitude
        self.store_and_fwd_flag = store_and_fwd_flag
        self.pickup_datetime = pickup_datetime

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "vendor_id": [self.vendor_id],
                "passenger_count": [self.passenger_count],
                "pickup_longitude": [self.pickup_longitude],
                "pickup_latitude": [self.pickup_latitude],
                "dropoff_longitude": [self.dropoff_longitude],
                "dropoff_latitude": [self.dropoff_latitude],
                "store_and_fwd_flag": [self.store_and_fwd_flag],
                "pickup_datetime": [self.pickup_datetime],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)