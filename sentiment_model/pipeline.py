from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentiment_model.processing.features import preprocess_text

def create_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, use_idf=True, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
