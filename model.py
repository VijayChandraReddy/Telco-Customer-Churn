import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load model hyperparameters from YAML."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_name: str, params: dict):
    """
    Train a classification model based on input model_name.
    
    Supported models: 'random_forest', 'gradient_boosting', 'svm', 'knn'
    """
    try:
        logger.debug("Training model: %s with params: %s", model_name, params)

        if model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"])
        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], random_state=params["random_state"])
        elif model_name == "svm":
            model = Pipeline([("scaler", StandardScaler()),
                              ("svm", SVC(C=params["C"], kernel=params["kernel"], random_state=params["random_state"], probability=True))])
        elif model_name == "knn":
            model = Pipeline([("scaler", StandardScaler()),
                              ("knn", KNeighborsClassifier(n_neighbors=params["n_neighbors"],weights = params['weights']))])
        else:
            raise ValueError("Unsupported model type. Choose from 'random_forest', 'gradient_boosting', 'svm', or 'knn'.")

        model.fit(X_train, y_train)
        logger.debug("%s model training completed.", model_name)
        return model
    except Exception as e:
        logger.error("Error training %s model: %s", model_name, e)
        raise

def save_model(model, file_path: str):
    """Save trained model as a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def main():
    try:
        # Load parameters (or set manually)
        params = {
            "random_forest": {"n_estimators": 25, "random_state": 2},
            "gradient_boosting": {"n_estimators": 50, "learning_rate": 0.1, "random_state": 2},
            "svm": {"C": 1.0, "kernel": "rbf", "random_state": 2},
            "knn": {"n_neighbors": 3 , 'weights' : 'distance'} 
        }

        # Load dataset
        train_data = load_data('./data/interim/train_processed.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        # Train and save models
        for model_name in ["random_forest", "gradient_boosting", "svm", "knn"]:
            model = train_model(X_train, y_train, model_name, params[model_name])
            save_model(model, f"models/{model_name}.pkl")

    except Exception as e:
        logger.error("Failed model-building process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()