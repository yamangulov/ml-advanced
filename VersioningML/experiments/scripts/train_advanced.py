import json
import os
import re

import mlflow
import mlflow.sklearn
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))


def load_jsonl(file_path, chunk_size=1000):
    """Загрузка данных из JSONL файла с постепенной обработкой"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            # Ограничиваем размер буфера для экономии памяти
            if len(data) >= chunk_size:
                yield pd.DataFrame(data)
                data = []
    
    # Обрабатываем оставшиеся данные
    if data:
        yield pd.DataFrame(data)


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
    with mlflow.start_run(run_name="advanced_model"):
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression_Optimized")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("preprocessing", "lowercase+punctuation_removal+stopwords_removal")

        # Load the data
        print("Loading data...")
        chunks = []
        total_size = 0
        for chunk in load_jsonl('data/kinopoisk.jsonl'):
            total_size += len(chunk)
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        chunks = None  # Освобождаем память

        # Log dataset info
        mlflow.log_param("dataset_size", total_size)
        mlflow.log_param("unique_labels", df['grade3'].nunique())

        # Preprocess the content
        print("Preprocessing text...")
        df['processed_content'] = df['content'].apply(preprocess_text)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_content'], df['grade3'], test_size=0.2, random_state=42, stratify=df['grade3']
        )

        # Create optimized pipeline based on baseline improvement
        print("Creating optimized pipeline...")
        
        # Use Logistic Regression with optimized parameters for speed and accuracy
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=12000,  # Оптимальное количество признаков
                ngram_range=(1, 2),  # 2-граммы для баланса скорости и качества
                min_df=2,            # Фильтрация редких слов
                max_df=0.95,         # Фильтрация слишком частых слов
                sublinear_tf=True    # Логарифмическое масштабирование
            )),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=4,
                class_weight='balanced'  # Учет дисбаланса классов
            ))
        ])

        # Log parameters
        mlflow.log_param("max_features", 12000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("min_df", 2)
        mlflow.log_param("max_df", 0.95)
        mlflow.log_param("sublinear_tf", True)
        mlflow.log_param("lr_C", 1.0)
        mlflow.log_param("lr_max_iter", 1000)
        mlflow.log_param("lr_random_state", 42)
        mlflow.log_param("lr_n_jobs", 4)
        mlflow.log_param("lr_class_weight", "balanced")

        # Hyperparameter tuning with GridSearchCV
        print("Performing hyperparameter tuning...")
        param_grid = {
            'tfidf__max_features': [10000, 12000],
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__min_df': [2, 3],
            'tfidf__max_df': [0.95],
            'classifier__C': [0.5, 1.0, 2.0]
        }

        # Use GridSearchCV for complete parameter search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3,
            scoring='accuracy', 
            n_jobs=4,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Make predictions
        y_pred = best_model.predict(X_test)

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

        # Cross-validation with fewer folds for speed
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy')
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
        mlflow.sklearn.log_model(best_model, "model")

        # Save additional artifacts
        os.makedirs("artifacts", exist_ok=True)

        # Save feature importance (for Logistic Regression - coefficients)
        feature_importance = np.abs(best_model.named_steps['classifier'].coef_[0])
        feature_names = best_model.named_steps['tfidf'].get_feature_names_out()

        # Get top 20 most important features
        top_indices = np.argsort(feature_importance)[-20:]
        top_features = feature_names[top_indices]
        top_importance = feature_importance[top_indices]

        importance_df = pd.DataFrame({
            'feature': top_features,
            'importance': top_importance
        }).sort_values('importance', ascending=False)

        importance_df.to_csv("artifacts/top_features.csv", index=False)
        mlflow.log_artifact("artifacts/top_features.csv")

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'text': X_test
        })
        predictions_df.to_csv("artifacts/predictions.csv", index=False)
        mlflow.log_artifact("artifacts/predictions.csv")

        # Save grid search results
        search_results = pd.DataFrame(grid_search.cv_results_)
        search_results.to_csv("artifacts/grid_search_results.csv", index=False)
        mlflow.log_artifact("artifacts/grid_search_results.csv")

        # Print results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
