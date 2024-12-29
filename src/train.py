import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint

from utils.selectors import LengthSelector, TextSelector


def optimize_pipeline(df, n_iter=20):
    param_distributions = {
        "features__title__tfidf__max_features": randint(50, 200),
        "features__text__tfidf__max_features": randint(100, 500),
        "classifier__max_iter": randint(500, 2000),
    }

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "title",
                            Pipeline(
                                [
                                    ("selector", TextSelector("title_clean")),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                                ]
                            ),
                        ),
                        (
                            "text",
                            Pipeline(
                                [
                                    ("selector", TextSelector("text_clean")),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                                ]
                            ),
                        ),
                        (
                            "title_length",
                            Pipeline(
                                [
                                    ("selector", LengthSelector("title_length")),
                                    ("scaler", RobustScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(df, df["fake"])

    print("\nBest parameters:", search.best_params_)
    print(f"\nBest CV accuracy: {search.best_score_:.3f}")

    return search.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )

    args = parser.parse_args()

    # Load data from SageMaker path
    df = pd.read_csv(os.path.join(args.data_dir, "processed_news.csv"))
    print(df.info())

    # Train and save model
    best_model = optimize_pipeline(df, n_iter=3)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(best_model, model_path)
