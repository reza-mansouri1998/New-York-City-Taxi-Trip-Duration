import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# Force Python to recognize the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ... now do your normal imports
from src.exception import CustomException


# Assuming your project includes standard custom logging and exception handling modules
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    """
    Configuration class to define the paths where the data artifacts will be saved.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw dataset, creates the artifacts folder, and splits the data into train and test sets.
        """
        logging.info("Entered the data ingestion component")
        try:
            # Load the NYC Taxi dataset
            file_path = r"D:\all\Python\ML Projects\New York City Taxi Trip Duration\notebook\data\train.csv"
            df = pd.read_csv(file_path)
            logging.info('Successfully read the dataset as a Pandas DataFrame')


            # NEW: Apply outlier filtering globally before splitting
            logging.info("Filtering target outliers globally")
            valid_idx = (df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)
            df = df[valid_idx].copy()
            

            # Create the artifacts directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts")

            # Perform the train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            # Return the paths for the next component in the pipeline (Data Transformation)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


# # Testing the component execution
# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()
#     print(f"Train data path: {train_data_path}")
#     print(f"Test data path: {test_data_path}")

#     data_transformation = DataTransformation()
#     # Fixed typo: initiate_data_transformation
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

#     modeltrainer = ModelTrainer()
#     # Fixed typo: initiate_model_training (so it doesn't crash on the next step)
#     modeltrainer.initiate_model_training(train_arr, test_arr)