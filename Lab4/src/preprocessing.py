from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer

import html
import numpy as np
import re

def clean_review(text):
    '''
    Làm sạch cột review:
    + Chuyển HTML entities về ký tự thông thường
    + Xóa HTML tags, như <br />
    + Chuẩn hóa khoảng trắng và chuyển văn bản về chữ thường
    '''
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()

def tokenize_review(text):
    """
    Chuyển review thành danh sách token.
    """
    text = clean_review(text)

    return re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
        text
    )

def build_tfidf_lr_pipeline(
    max_features = 10000,
    ngram_range = (1, 1),
    min_df=1,
    random_state = 2026
):
    '''
    Thiết lập pipeline xử lý TF-TDF và Logistic Regression
    Parameter:
    + max_features: Số feature TF-IDF tối đa
    + ngram_range: (1, 1) cho unigram, (1, 2) cho unigram + bigram
    + random_state: seed để kết quả có thể tái lập
    + min_df: loại các từ xuất hiện ít theo review
    Output: Pipeline chưa được fit
    '''
    return Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                preprocessor = clean_review,
                lowercase = False,
                max_features = max_features,
                ngram_range = ngram_range,
                min_df = min_df,
                stop_words = None
            )
        ),
        (
            'classifier',
            LogisticRegression(
                max_iter = 1000,
                random_state = random_state
            )
        )
    ])

def get_vocabulary_size(model):
    '''
    Trả về kích thước vocabulary sau khi model đã fit
    '''
    tfidf = model.named_steps['tfidf']
    check_is_fitted(tfidf, 'vocabulary_')

    return len(tfidf.vocabulary_)

def split_dataset(
    data,
    text_column = 'review',
    label_column = 'sentiment',
    train_size = 0.6,
    val_size = 0.2,
    test_size = 0.2,
    random_state = 2026
):
    '''
    Chia dataset thành ba bộ train, validation, test
    Mặc định: Train 60%, Validation 20%, Test 20%
    '''
    total_size = train_size + val_size + test_size

    if not abs(total_size - 1.0) < 1e-9:
        raise ValueError(
            "Tổng train_size, validation_size và test_size phải bằng 1."
        )

    required_columns = {text_column, label_column}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f'Dataset phải có các cột: {required_columns}'
        )

    X = data[text_column]
    y = data[label_column]

    # Chia 60% train và 40% còn lại
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    # Chia 40% còn lại thành 20% validation và 20% test
    relative_test_size = test_size / (
        val_size + test_size
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_average_word2vec_features(texts, model):
    """
    Biểu diễn mỗi document bằng trung bình vector của những từ có trong vocabulary Word2Vec.
    Nếu document không có từ nào trong vocabulary, trả về vector 0.
    """
    document_vectors = []
    vector_size = model.wv.vector_size

    for text in texts:
        tokens = tokenize_review(text)

        known_vectors = [
            model.wv[token]
            for token in tokens
            if token in model.wv
        ]

        if known_vectors:
            document_vector = np.mean(
                known_vectors,
                axis=0
            )
        else:
            document_vector = np.zeros(
                vector_size,
                dtype=np.float32
            )

        document_vectors.append(document_vector)

    return np.vstack(document_vectors)