import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer, BertModel
import torch
from typing import Tuple, Dict, Callable, Any

# Завантаження та підготовка даних
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)
X = newsgroups.data
y = newsgroups.target

# Поділ на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Методи трансформації тексту
def tfidf_transform(X_train: list[str], X_test: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Трансформація тексту у TF-IDF вектори.

    :param X_train: Список текстів навчальної вибірки.
    :param X_test: Список текстів тестової вибірки.
    :return: Трансформовані навчальні та тестові дані у вигляді TF-IDF векторів.
    """
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

def word2vec_transform(X_train: list[str], X_test: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Трансформація тексту у Word2Vec вектори.

    :param X_train: Список текстів навчальної вибірки.
    :param X_test: Список текстів тестової вибірки.
    :return: Трансформовані навчальні та тестові дані у вигляді Word2Vec векторів.
    """
    model = gensim.models.Word2Vec([text.split() for text in X_train], vector_size=100, window=5, min_count=1, workers=4)
    X_train_w2v = np.array([np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(100)], axis=0) for text in X_train])
    X_test_w2v = np.array([np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(100)], axis=0) for text in X_test])
    return X_train_w2v, X_test_w2v

def doc2vec_transform(X_train: list[str], X_test: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Трансформація тексту у Doc2Vec вектори.

    :param X_train: Список текстів навчальної вибірки.
    :param X_test: Список текстів тестової вибірки.
    :return: Трансформовані навчальні та тестові дані у вигляді Doc2Vec векторів.
    """
    tagged_data = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(X_train)]
    model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4)
    X_train_d2v = np.array([model.infer_vector(text.split()) for text in X_train])
    X_test_d2v = np.array([model.infer_vector(text.split()) for text in X_test])
    return X_train_d2v, X_test_d2v

def transformer_embeddings(X_train: list[str], X_test: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Трансформація тексту у векторні подання за допомогою BERT.

    :param X_train: Список текстів навчальної вибірки.
    :param X_test: Список текстів тестової вибірки.
    :return: Трансформовані навчальні та тестові дані у вигляді BERT векторів.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def encode(texts: list[str]) -> np.ndarray:
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    X_train_bert = encode(X_train)
    X_test_bert = encode(X_test)
    return X_train_bert, X_test_bert

# Функція для оцінки моделей
def evaluate_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Оцінка моделей за допомогою перехресної валідації.

    :param models: Словник моделей для оцінки.
    :param X_train: Трансформовані навчальні дані.
    :param y_train: Мітки навчальних даних.
    :return: Словник з назвами моделей та їх середніми оцінками і стандартними відхиленнями.
    """
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = (np.mean(scores), np.std(scores))
        print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return results

# Список моделей для оцінки
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Методи трансформації тексту для оцінки
transformation_methods = {
    "TF-IDF": tfidf_transform,
    "Word2Vec": word2vec_transform,
    "Doc2Vec": doc2vec_transform,
    "BERT Embeddings": transformer_embeddings
}

# Оцінка моделей для різних способів трансформації тексту
for method_name, transform in transformation_methods.items():
    print(f"\nОцінка моделей для методу трансформації: {method_name}")
    
    # Трансформація даних
    X_train_transformed, X_test_transformed = transform(X_train, X_test)
    
    # Оцінка моделей
    results = evaluate_models(models, X_train_transformed, y_train)
    
    # Вибір найкращої моделі для поточного методу трансформації
    best_model_name = max(results, key=lambda k: results[k][0])
    best_model = models[best_model_name]
    print(f"\nНайкраща модель для {method_name}: {best_model_name} з точністю {results[best_model_name][0]:.4f}")
    
    # Навчання та оцінка найкращої моделі на тестовій вибірці
    best_model.fit(X_train_transformed, y_train)
    y_pred_train = best_model.predict(X_train_transformed)
    y_pred_test = best_model.predict(X_test_transformed)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Точність на навчальній вибірці для {method_name}: {train_accuracy:.4f}")
    print(f"Точність на тестовій вибірці для {method_name}: {test_accuracy:.4f}")
    
    # Візуалізація тренувальних і тестових помилок
    plt.figure(figsize=(10, 5))
    plt.bar(['Train Error', 'Test Error'], [1 - train_accuracy, 1 - test_accuracy], color=['blue', 'orange'])
    plt.title(f'Навчальні і тестові помилки для {best_model_name} з використанням {method_name}')
    plt.ylabel('Помилка')
    plt.ylim(0, 1)
    plt.show()
