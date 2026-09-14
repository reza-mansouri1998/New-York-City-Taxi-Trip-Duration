import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """
        Executes the end-to-end training process by instantiating and 
        running the components sequentially.
        """
        try:
            logging.info("=== Starting the End-to-End Training Pipeline ===")
            
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Step 1: Data Ingestion completed")

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Step 2: Data Transformation completed")

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            test_rmsle = model_trainer.initiate_model_training(train_arr, test_arr)
            logging.info(f"Step 3: Model Training completed. Final Test RMSLE: {test_rmsle:.4f}")
            
            logging.info("=== Training Pipeline successfully finished ===")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()