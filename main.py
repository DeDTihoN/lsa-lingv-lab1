import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy.linalg import norm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import nltk


def svd(X):
    # Обчислення сингулярного розкладу
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    return U, S, VT


def get_word_topic(word, terms, U_k):
    # Знаходимо індекс слова у словнику
    word_index = list(terms).index(word)
    if word_index == -1:
        return -1

    topic_number = -1
    for i in range(len(U_k[word_index, :])):
        if topic_number == -1 or abs(U_k[word_index, i]) > abs(U_k[word_index, topic_number]):
            topic_number = i

    return topic_number


def get_topic_top_words(num, terms, count, U_k):
    return [terms[i] for i in np.argsort(-abs(U_k[:, num]))[:count]]


def get_topic_name(num, terms, U_k):
    return get_topic_top_words(num, terms, 1, U_k)[0]


def get_document_topic(doc_number, V_t):
    # Знаходимо індекс документу у словнику

    topic_number = -1
    for i in range(len(V_t)):
        if topic_number == -1 or abs(V_t[i, doc_number]) > abs(V_t[topic_number, doc_number]):
            topic_number = i

    return topic_number


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def find_closest_word(word, terms, U_k):
    word_topic = get_word_topic(word, terms, U_k)
    max_similarity = -1
    closest_word = None

    for i, term in enumerate(terms):
        if term == word:
            continue

        term_vector = U_k[i, :]
        similarity = cosine_similarity(word_topic, term_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            closest_word = term

    return closest_word


# Завантажуємо ресурси NLTK
# nltk.download('punkt')
# nltk.download('punkt_tab')
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


with open('test.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Розділяємо текст на окремі документи за допомогою подвійного пробілу
documents = content.split('\n')

# print(documents)

# Обробка документів
processed_documents = [preprocess_text(doc) for doc in documents]

# Створюємо матрицю частот слів з видаленням стоп-слів
vectorizer = TfidfVectorizer(stop_words='english')  # Використання стандартних англійських стоп-слів
X = vectorizer.fit_transform(processed_documents).toarray().T  # Транспонуємо матрицю, щоб отримати (слова, документи)

# Виведення результатів
terms = vectorizer.get_feature_names_out()

# Викликаємо SVD на матриці частот
U, S, VT = svd(X)

# Вибираємо кількість компонент для зменшення
k = min(len(documents), 3)  # Залишаємо дві головні теми

# Зменшуємо розмірність, залишивши тільки k компонент
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

print("VT_k: ", VT_k)

# Відновлюємо зменшену матрицю
X_reduced = np.dot(U_k, np.dot(S_k, VT_k))

for i, doc in enumerate(documents):
    topic = get_document_topic(i, VT_k)
    print(f"Document: {i}, Topic: {get_topic_name(topic, terms, U_k)}")

for i in range(k):
    top10_words = get_topic_top_words(i, terms, 10, U_k)
    print(f"Topic {i}: {', '.join(top10_words)}")

# for word in terms:
#     topic = get_word_topic(word, terms, U_k)
#     print(f"Word: {word}, Topic: {get_topic_name(topic, terms, U_k)}")
