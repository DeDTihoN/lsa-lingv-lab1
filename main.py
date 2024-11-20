import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans

# Завантажуємо ресурси NLTK
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Ініціалізуємо лемматизатор та стоп-слова
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Функція для лемматизації та видалення стоп-слів
def preprocess_text(text):
    # Токенізація
    words = word_tokenize(text.lower())
    # Лемматизація та видалення стоп-слів
    return ' '.join(lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words)


# Завантажуємо документи з набору 20 Newsgroups
categories = ['rec.sport.baseball', 'comp.graphics', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# Розділяємо документи по категоріях, щоб отримати рівномірний розподіл документів
category_docs = {category: [] for category in categories}
for doc, target in zip(newsgroups.data, newsgroups.target):
    category_name = newsgroups.target_names[target]
    if category_name in category_docs:
        category_docs[category_name].append(doc)

# Переконуємося, що в кожній категорії достатньо документів
documents_per_category = 100  # Збільшимо кількість документів для кожної категорії для кращої кластеризації
selected_documents = []
for category in categories:
    if len(category_docs[category]) >= documents_per_category:
        selected_documents.extend(category_docs[category][:documents_per_category])
    else:
        print(f"Warning: Not enough documents in category '{category}' (found {len(category_docs[category])})")
        selected_documents.extend(category_docs[category])

# Обробка документів
processed_documents = [preprocess_text(doc) for doc in selected_documents]

# Створюємо матрицю частот слів з видаленням стоп-слів
vectorizer = TfidfVectorizer(stop_words='english')  # Використання стандартних англійських стоп-слів
X = vectorizer.fit_transform(processed_documents)  # Трансформуємо документи у матрицю частот слів
terms = vectorizer.get_feature_names_out()

# Виконуємо SVD для зменшення розмірності
k = 3  # Кількість тем для отримання
svd = TruncatedSVD(n_components=k)
X_reduced = svd.fit_transform(X)

# Виконуємо кластеризацію документів за допомогою KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_reduced)
labels = kmeans.labels_

# Виводимо результати кластеризації документів
for i, label in enumerate(labels):
    print(f"Document {i + 1}: Cluster {label + 1}")

# Виконуємо кластеризацію слів за допомогою KMeans
kmeans_words = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_words.fit(svd.components_.T)
word_labels = kmeans_words.labels_

# Виводимо результати кластеризації слів
for cluster in range(n_clusters):
    cluster_terms_indices = [i for i, label in enumerate(word_labels) if label == cluster]
    sorted_indices = sorted(cluster_terms_indices, key=lambda x: -abs(svd.components_[:, x]).sum())
    top_terms = [terms[index] for index in sorted_indices[:10]]
    print(f"Word Cluster {cluster + 1}: {', '.join(top_terms)}")
