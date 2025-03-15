import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dvclive import Live

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading data: %s", e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise

def load_model(file_path: str):
    """Load the trained model from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", file_path, e)
        raise
    
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        if np.any(np.isin(y_test, ['Yes', 'No'])):
            y_test_numeric = np.where(y_test == 'Yes', 1, 0)
        else:
            if not np.issubdtype(y_test.dtype, np.number):
                logger.error("Unexpected value in y_test, not 'Yes'/'No' or numeric.")
                raise ValueError("y_test contains invalid values.")
            y_test_numeric = y_test

        y_pred = clf.predict(X_test)

        # Check if 'predict_proba' is available
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
            logger.warning("Model does not support probability predictions.")

        accuracy = accuracy_score(y_test_numeric, y_pred)
        precision = precision_score(y_test_numeric, y_pred)
        recall = recall_score(y_test_numeric, y_pred)

        # Only calculate AUC if probabilities are available
        auc = roc_auc_score(y_test_numeric, y_pred_proba) if y_pred_proba is not None else None

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
def main():
    try:
        # Load the test data
        test_data = load_data('./data/interim/test_processed.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.debug(f"Test data shape: {X_test.shape}")

        # Load models and evaluate
        models = ['random_forest', 'gradient_boosting', 'svm', 'knn']
        best_model = None
        best_score = float('-inf')
        best_model_name = None

        for model_name in models:
            model = load_model(f"models/{model_name}.pkl")
            results = evaluate_model(model, X_test, y_test)

            # Save metrics
            save_metrics(results, f'reports/{model_name}_metrics.json')

            # Predict and store results
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Save predictions along with actual labels
            output_df = pd.DataFrame(X_test, columns=test_data.columns[:-1])  # Use same column names for features
            output_df["Actual"] = y_test
            output_df["Predicted"] = y_pred

            if y_pred_proba is not None:
                output_df["Predicted_Probability"] = y_pred_proba

            output_file = f'predictions/{model_name}_predictions.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")

            # Choose the best model based on AUC (or accuracy if AUC is None)
            model_score = results['auc'] if results['auc'] is not None else results['accuracy']
            if model_score > best_score:
                best_score = model_score
                best_model = model
                best_model_name = model_name

        # Save the best model
        if best_model:
            best_model_path = f"models/best_model.pkl"
            with open(best_model_path, "wb") as file:
                pickle.dump(best_model, file)
            logger.info(f"Best model ({best_model_name}) saved to {best_model_path}")

    except Exception as e:
        logger.error("Error in model evaluation process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

