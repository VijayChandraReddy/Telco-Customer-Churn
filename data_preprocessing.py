import os
import logging
import pandas as pd
import numpy as np
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Ensure "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_cat(df, encoder=None):
    """
    Encodes categorical features using OneHotEncoder and renames Churn column properly.
    This function accepts an optional pre-fitted encoder for consistency between training and test data.
    """
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    
    if len(categorical_cols) == 0:
        logger.debug('No categorical columns found for encoding.')
        return df

    if encoder is None:
        encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
        encoder.fit(df[categorical_cols])  # Fit on training data (only once)

    encoded_data = encoder.transform(df[categorical_cols])  # Transform data based on the fitted encoder

    if encoded_data.shape[1] == 0:
        logger.warning("No categorical variables were successfully encoded.")
        return df  # Return original df if nothing was encoded

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    df_transformed = df.drop(columns=categorical_cols, errors='ignore').reset_index(drop=True)
    df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

    if 'Churn_No' in df_transformed.columns:
        df_transformed.drop('Churn_No', axis=1, inplace=True)
    if 'Churn_Yes' in df_transformed.columns:
        df_transformed.rename(columns={'Churn_Yes': 'Churn'}, inplace=True)

    df_transformed.fillna(0, inplace=True)  # Handle missing values
    return df_transformed, encoder




def preprocess_df(df, encoder=None):
    """
    Removes duplicates and applies categorical transformation.
    Optionally returns the encoder to be used on the test set.
    """
    try:
        if df.empty:
            logger.warning('Empty DataFrame detected.')
            return df, encoder
        
        original_shape = df.shape
        df = df.drop_duplicates(keep='first')
        logger.debug(f'Duplicates removed: {original_shape[0] - df.shape[0]} rows')
        
        df, encoder = transform_cat(df, encoder)
        return df, encoder
    except Exception as e:
        logger.error(f'Error in preprocessing: {e}')
        raise


def visualize_data(df):
    """
    Function to create visualizations of the data (e.g., distribution of features, correlations).
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

        # Fixing correlation matrix
        plt.figure(figsize=(10, 6))
        corr_matrix = df.corr().fillna(0)  # Replace NaN values with 0
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

        logger.debug('Visualizations created')
    except Exception as e:
        logger.error(f'Error during data visualization: {e}')
        raise


def main():
    """
    Loads raw data, preprocesses it, visualizes it, and saves processed data.
    """
    try:
        raw_data_path = './data/raw'
        train_file = os.path.join(raw_data_path, 'train_data')
        test_file = os.path.join(raw_data_path, 'test_data')

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        logger.debug('Data loaded successfully')

        # Preprocessing data and getting encoder from training set
        train_processed_data, encoder = preprocess_df(train_data)
        test_processed_data, _ = preprocess_df(test_data, encoder)

        print(f"Train shape: {train_processed_data.shape}")
        print(f"Test shape: {test_processed_data.shape}")

        # Visualizing data (optional)
        # visualize_data(train_processed_data)
        # visualize_data(test_processed_data)

        # Save processed data
        processed_path = './data/interim'
        os.makedirs(processed_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)

        logger.debug(f'Processed data saved in {processed_path}')
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except pd.errors.EmptyDataError as e:
        logger.error(f'No data: {e}')
    except Exception as e:
        logger.error(f'Error in main processing: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
