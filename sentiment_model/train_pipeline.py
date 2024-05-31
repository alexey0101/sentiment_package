import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sentiment_model.config.core import fetch_config_from_yaml
from sentiment_model.pipeline import create_pipeline
from sentiment_model.processing.features import preprocess_text  # Import the preprocess_text function
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = fetch_config_from_yaml('sentiment_model/config.yml')

def train():
    logger.info("Starting training process.")
    # Read the dataset
    data = pd.read_csv(config.data.dataset_path, encoding='ISO-8859-1', header=None)
    data = data.drop(columns=0)
    data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

    positive_tweets = data[data['target'] == 4].sample(n=config.training.samples // 2, random_state=42)
    negative_tweets = data[data['target'] == 0].sample(n=config.training.samples // 2, random_state=42)
    balanced_data = pd.concat([positive_tweets, negative_tweets])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Preprocess the data
    balanced_data['text'] = balanced_data['text'].apply(preprocess_text)

    X = balanced_data['text']
    y = balanced_data['target'].replace({0: 0, 4: 1})

    logger.info("Splitting the data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.data.test_size, random_state=42)

    logger.info("Creating and training the pipeline.")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Define the directory and file name
    directory = 'sentiment_model/trained_models'
    file_name = f"{config.model.name}_{config.model.version}.pkl"
    file_path = os.path.join(directory, file_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.info("Saving the trained model.")
    dump(pipeline, file_path)
    logger.info(f"Model trained and saved as {file_path}")

if __name__ == "__main__":
    train()
