import os
import sys
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object (like a machine learning model or preprocessor) 
    to a specified file path as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Loads and returns a pickle file from the given path."""
    import pickle
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        import sys
        from src.exception import CustomException
        raise CustomException(e, sys)