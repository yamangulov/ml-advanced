## 11.6 Практическая работа
### Цель практической работы
Научиться:

- самостоятельно проектировать эксперимент,
- выбирать метрики и показатели для трекинга во время эксперимента,
- запускать эксперименты на MLflow.

### Что нужно сделать
Используйте [датасет](kinopoisk.jsonl).

Вы получили от коллеги скрипт, который нужно привести в порядок. Это скрипт, который тренирует самую простую ML-модель на датасете (вы решаете задачу классификации отзывов о фильме: позитивный отзыв или негативный). Только он не соответствует лучшим практикам работы с ML-артефактами. Где полученная модель — непонятно, какие у неё метрики — тоже.

Ваша задача — привести скрипт в порядок:

- Установить зависимости для скрипта:
```
pip install pandas numpy scikit-learn nltk
```
- Добавить его в репозиторий из предыдущей задачи.
- Добавить в скрипт MLflow Tracking для логирования результатов и провести эксперимент.
- Добавить ещё один скрипт с альтернативной моделью (любой на ваше усмотрение) и запустить его.
- Сравнить результаты двух экспериментов в MLflow web UI, предоставить скриншоты и указать, какая модель показывает лучшие результаты на выбранных метриках.
- Закоммитить все обновлённые скрипты, артефакты моделей и логи их запусков.

**Скрипт:**
```
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))


# Load the JSONL file
def load_jsonl(file_path):
   data = []
   with open(file_path, 'r', encoding='utf-8') as f:
       for line in f:
           data.append(json.loads(line))
   return pd.DataFrame(data)

# Preprocess text
def preprocess_text(text):
   # Convert to lowercase
   text = text.lower()
   # Remove punctuation
   text = re.sub(r'[^\w\s]', '', text)
   # Remove stopwords
   words = text.split()
   words = [word for word in words if word not in russian_stopwords]
   return ' '.join(words)

# Load the data
df = load_jsonl('kinopoisk.jsonl')

# Preprocess the content
df['processed_content'] = df['content'].apply(preprocess_text)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
   df['processed_content'], df['grade3'], test_size=0.2, random_state=42
)

# Create bag of words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

# Make predictions
y_pred = clf.predict(X_test_bow)

# Evaluate the model
print(classification_report(y_test, y_pred))


# Function to classify new reviews
def classify_review(review_text):
   processed_text = preprocess_text(review_text)
   bow_representation = vectorizer.transform([processed_text])
   prediction = clf.predict(bow_representation)
   return prediction[0]

# Example usage
#example_review = "Ваш текст рецензии на русском языке"
#print(f"Классификация: {classify_review(example_review)}")
```

1. Склонируйте изначальный проект с моделью.
2. Имплементируйте поверх текущей модели MLflow.
3. Запустите код и получите изначальные метрики в MLflow.
4. Используйте любую другую модель, чтобы улучшить выбранную метрику.
5. Сравните результаты и закоммитьте обновлённый скрипт с моделью в репозиторий.

### Критерии оценки
**Принято:**

- Добавлен трекинг MLflow.
- На скриншоте видны обе модели.
- В тренировочном скрипте tfidf заменена на лучшую модель.
- Метрика модели улучшилась.

**На доработку:** один из критериев выше не соблюдён.