import os
import sys
from dataclasses import dataclass
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import root_mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    """Configuration for saving the trained model object."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Splits the transformed arrays, trains the optimized LightGBM model, 
        evaluates it, and saves the model artifact.
        """
        try:
            logging.info("Splitting transformed data into dependent and independent variables")
            # In np.c_[X, y], the last column is the target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Initializing the optimized LightGBM model")
            # These are the exact optimal parameters discovered during notebook tuning
            model = LGBMRegressor(
                subsample=1.0,
                reg_lambda=15,
                reg_alpha=2.0,
                num_leaves=127,
                min_child_samples=50,
                max_depth=9,
                learning_rate=0.08,
                colsample_bytree=1.0,
                n_estimators=4000,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )

            logging.info("Training the model with early stopping")
            # Passing eval_set to dynamically halt training and prevent overfitting
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)], # Keep this for standard API compatibility
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
            )

            logging.info("Saving the final trained model to artifacts")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Evaluating final model performance")
            predicted = model.predict(X_test)
            
            # The target is log-transformed, so standard RMSE here equals Kaggle's RMSLE
            rmsle = root_mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)
            
            logging.info(f"Final Model Test RMSLE: {rmsle:.4f}")
            logging.info(f"Final Model Test R2: {r2:.4f}")

            return rmsle

        except Exception as e:
            raise CustomException(e, sys)