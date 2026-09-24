import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from implementation import TF_IDF_Vectorizer

# Corpus kiểm thử
corpus = [
    "cat eats fish",
    "dog eats fish",
    "cat likes fish"
]

# Chương trình tự xây

my_vectorizer = TF_IDF_Vectorizer(corpus)

my_vocab = my_vectorizer.build_vocabulary()
my_idf = my_vectorizer.compute_idf()
my_tfidf_dict = my_vectorizer.compute_tfidf()

sklearn_vectorizer = TfidfVectorizer(
    lowercase=True,
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    use_idf=True,
    smooth_idf=False,
    sublinear_tf=False,
    norm=None
)

# Thư viện

X = sklearn_vectorizer.fit_transform(corpus)

sklearn_vocab = sklearn_vectorizer.get_feature_names_out()
sklearn_idf = sklearn_vectorizer.idf_

# So sánh bộ term
print("My vocabulary:")
print(my_vocab)

print("\nSklearn vocabulary:")
print(list(sklearn_vocab))


# So sánh IDF
my_idf_array = np.array([
    my_idf[term]
    for term in sklearn_vocab
])

print("My IDF:")
print(my_idf_array)

print("\nSklearn IDF:")
print(sklearn_idf)

print("\nIDF equal:",
      np.allclose(my_idf_array, sklearn_idf))


# So sánh TF-IDF
my_tfidf = np.array([
    [
        my_tfidf_dict[doc_id][term]
        for term in sklearn_vocab
    ]
    for doc_id in range(len(corpus))
])

sklearn_tfidf = X.toarray()

print("My TF-IDF:")
print(my_tfidf)

print("\nSklearn TF-IDF:")
print(sklearn_tfidf)

print(
    "TF-IDF equal:",
    np.allclose(my_tfidf, sklearn_tfidf)
)


# So sánh cosine similarity giữa 2 vector đầu tiên
my_cosine = my_vectorizer.cosine_similarity(
    my_tfidf[0],
    my_tfidf[1]
)

sklearn_cosine = cosine_similarity(
    X[0],
    X[1]
)[0, 0]

print("My cosine:", my_cosine)
print("Sklearn cosine:", sklearn_cosine)

# Khác biệt giữa chương trình tự xây dựng và thư viện nằm ở công thức
# + TF: Tự xây dựng: tf(t, d) = count(term, document) / (tổng số term trong document)
#          Thư viện: tf(t, d) = count(term, document)
# + IDF: Tự xây dựng: idf(t, d) = ln( N / DF )
#          Thư viện: idf(t, d) = ln( N / DF ) + 1
# Công thức cosine similarity vẫn như nhau ở cả hai chương trình
# Nhờ đó mà kết quả TF-IDF và cosine similarity có sự khác biệt