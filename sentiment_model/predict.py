import pandas as pd
from joblib import load
from sentiment_model.config.core import fetch_config_from_yaml

config = fetch_config_from_yaml('sentiment_model/config.yml')
model = load(f"{config.model.name}_{config.model.version}.pkl")

def predict(texts):
    predictions = model.predict(texts)
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [sentiment_map[pred] for pred in predictions]

if __name__ == "__main__":
    sample_texts = ["I love this!", "I hate this...", "This is okay."]
    preds = predict(sample_texts)
    print(preds)