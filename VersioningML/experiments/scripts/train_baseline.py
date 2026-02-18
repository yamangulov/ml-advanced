import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
import re
import mlflow
import mlflow.sklearn
import os

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))


def load_jsonl(file_path):
    """Загрузка данных из JSONL файла"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def preprocess_text(text):
    """Предобработка текста"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)


def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")

    # Start MLflow run
    with mlflow.start_run(run_name="baseline_model"):
        # Log parameters
        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("preprocessing", "lowercase+punctuation_removal+stopwords_removal")

        # Load the data
        print("Loading data...")
        df = load_jsonl('data/kinopoisk.jsonl')

        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("unique_labels", df['grade3'].nunique())

        # Preprocess the content
        print("Preprocessing text...")
        df['processed_content'] = df['content'].apply(preprocess_text)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_content'], df['grade3'], test_size=0.2, random_state=42, stratify=df['grade3']
        )

        # Create TF-IDF representation (улучшение baseline)
        print("Creating TF-IDF representation...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Log vectorizer parameters
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("min_df", 2)
        mlflow.log_param("max_df", 0.95)

        # Train the classifier
        print("Training model...")
        clf = MultinomialNB(alpha=1.0)
        clf.fit(X_train_tfidf, y_train)

        # Log model parameters
        mlflow.log_param("alpha", 1.0)

        # Make predictions
        y_pred = clf.predict(X_test_tfidf)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"{label}_precision", metrics['precision'])
                mlflow.log_metric(f"{label}_recall", metrics['recall'])
                mlflow.log_metric(f"{label}_f1-score", metrics['f1-score'])
                mlflow.log_metric(f"{label}_support", metrics['support'])

        # Log the model
        mlflow.sklearn.log_model(clf, "model")
        mlflow.sklearn.log_model(vectorizer, "vectorizer")

        # Save additional artifacts
        os.makedirs("artifacts", exist_ok=True)

        # Save feature names
        feature_names = vectorizer.get_feature_names_out()
        np.save("artifacts/feature_names.npy", feature_names)
        mlflow.log_artifact("artifacts/feature_names.npy")

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'text': X_test
        })
        predictions_df.to_csv("artifacts/predictions.csv", index=False)
        mlflow.log_artifact("artifacts/predictions.csv")

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()