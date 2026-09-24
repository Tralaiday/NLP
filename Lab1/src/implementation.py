import numpy as np

class TF_IDF_Vectorizer:

    # Hàm dựng với đầu vào là corpus (tập văn bản) cần xử lý
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab_list = None

    # Hàm xây dựng tập vocabulary/term xuất hiện trong văn bản
    # Output: List chứa các vocab/term xuất hiện trên các document trong corpus
    # === AI contribution ===
    # Gợi ý sử dụng cấu trúc dữ liệu set để lưu trữ các term sao cho không bị duplicated
    # Hỗ trợ chuyển set các term về dạng sorted list để dễ lưu trữ, truy xuất và tính toán
    def build_vocabulary(self):
        vocab_set = set()

        for document in self.corpus:
            tokens = document.lower().split()

            for token in tokens:
                vocab_set.add(token)

        self.vocab_list = sorted(vocab_set)
        return self.vocab_list

    # Hàm tính count vector của các document trong corpus
    # Output: list chứa các count vector với document tương ứng
    # === AI contribution ===
    # Gợi ý tách document thành list các token in thường
    # Xử lý việc đếm số lần term xuất hiện trong document
    def compute_counts(self):
        count_vector_list = []
        num_vocab = len(self.vocab_list)

        for document in self.corpus:
            count_vector_doc = [0] * num_vocab
            doc_split_list = document.lower().split()

            for i in range(num_vocab):
                count_vector_doc[i] = doc_split_list.count(self.vocab_list[i])

            count_vector_list.append(count_vector_doc)
        return count_vector_list

    # Hàm tính TF của các term theo từng document trong corpus
    # Output: Dictionary chứa giá trị TF của term trong document chỉ số i tương ứng ~ TF matrix của corpus
    # Ví dụ: {0: {'cat': 0.33333, 'dog': 0.66666, ...}, .... }
    # === AI contribution ===
    # Gợi ý áp dụng count vector representation để tính toán TF của từng term
    # Lấy key dict chứa giá trị TF là term trong vocab_list
    # Sửa phần total_terms là tổng số term trong count_vector, không phải độ dài count_vector
    def compute_tf(self):
        tf_dict = {}
        count_vectors = self.compute_counts()

        for i, vector in enumerate(count_vectors):
            tf_document = {}
            total_terms = sum(vector)

            for j, term_count in enumerate(vector):
                term = self.vocab_list[j]
                tf = term_count / total_terms
                tf_document[term] = tf

            tf_dict[i] = tf_document

        return tf_dict

    # Hàm tính DF (số document chứa term bất kỳ)
    # Output: dict chứa các giá trị DF của từng term ~ vector DF của vocab_list
    # === AI contribution ==
    # Lưu ý về truy xuất chỉ số j ứng với term tương ứng
    def __compute_df(self):
        count_vector_list = self.compute_counts()
        df_dict = {}

        for j, term in enumerate(self.vocab_list):
            document_count = 0

            for count_vector in count_vector_list:
                if count_vector[j] > 0:
                    document_count += 1

            df_dict[term] = document_count

        return df_dict

    # Hàm tính vector IDF
    # Output: dict chứa các giá trị IDF của từng term ~ vector IDF của vocab_list
    def compute_idf(self):
        num_documents = len(self.corpus)
        df_dict = self.__compute_df()
        idf_dict = {}

        for term, df in df_dict.items():
            idf_dict[term] = np.log(num_documents / df)

        return idf_dict

    # Hàm tính ma trận TF-IDF (số hàng là số document, số cột là số term trong vocab_list)
    # Output: Dictionary chứa giá trị TF-IDF của term trong document chỉ số i tương ứng ~ TF-IDF matrix của corpus
    # === AI contribution ===
    # Cải tiến về term key in dict con của tf_idf_dict
    def compute_tfidf(self):
        tf_dict = self.compute_tf()
        idf_dict = self.compute_idf()
        tf_idf_dict = {}

        for i in tf_dict.keys():
            tf_idf_document = {}

            for term, idf in idf_dict.items():
                tf = tf_dict[i][term]
                tf_idf_document[term] = tf * idf

            tf_idf_dict[i] = tf_idf_document

        return tf_idf_dict

    # Hàm tính cosine similarity giữa 2 vector
    # Output: cosine similarity giữa 2 vector
    # === AI contribution ===
    # Gợi ý cách sử dụng thư viện NumPy để xây dựng công thức cosine similarity
    # Lưu ý về trường hợp đặc biệt khi chuẩn của 1 trong 2 vector đang xét bằng 0
    def cosine_similarity(self, vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        dot_product = np.dot(vector1, vector2)
        
        vec1_norm = np.linalg.norm(vector1)
        vec2_norm = np.linalg.norm(vector2)

        if vec1_norm == 0 or vec2_norm == 0:
            return 0.0

        return dot_product / (vec1_norm * vec2_norm)

# Chương trình kiểm thử
if __name__ == '__main__':
    # Corpus kiểm thử
    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    # Khai báo đối tượng vectorizer và nhập corpus kiểm thử để tiến hành xử lý
    vectorizer = TF_IDF_Vectorizer(corpus)

    # Unit test cho hàm build_vocabulary()
    vocab_list = vectorizer.build_vocabulary()
    assert vocab_list == ['cat', 'dog', 'eats', 'fish', 'likes']

    # Unit test cho hàm compute_counts()
    count_vector_list = vectorizer.compute_counts()
    assert count_vector_list == [
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1]
    ]

    # Unit test cho hàm compute_tf(), với N x V unit test (N: số document, V: số term), cụ thể là 15 unit test
    tf_dict = vectorizer.compute_tf()
    expected_tf_dict = {
        0: {'cat': 1 / 3, 'dog': 0, 'eats': 1 / 3,  'fish': 1 / 3, 'likes': 0},
        1: {'cat': 0, 'dog': 1 / 3, 'eats': 1 / 3,  'fish': 1 / 3, 'likes': 0},
        2: {'cat': 1 / 3, 'dog': 0, 'eats': 0,  'fish': 1 / 3, 'likes': 1 / 3}
    }

    for doc_id in expected_tf_dict.keys():
        for term in expected_tf_dict[doc_id].keys():
            assert abs(tf_dict[doc_id][term] - expected_tf_dict[doc_id][term]) < 1e-10

    # Unit test cho hàm compute_idf(), với V unit test, cụ thể là 5 unit test
    idf_dict = vectorizer.compute_idf()
    expected_idf_dict = {
        'cat': np.log(3 / 2),
        'dog': np.log(3),
        'eats': np.log(3 / 2),
        'fish': 0,
        'likes': np.log(3)
    }

    for term in expected_idf_dict.keys():
        assert abs(expected_idf_dict[term] - idf_dict[term]) < 1e-10

    # Unit test cho hàm compute_tfidf(), với N x V unit test (N: số document, V: số term), cụ thể là 15 unit test
    tf_idf_dict = vectorizer.compute_tfidf()
    expected_tf_idf_dict = {
        0: {'cat': np.log(3 / 2) / 3, 'dog': 0, 'eats': np.log(3 / 2) / 3,  'fish': 0, 'likes': 0},
        1: {'cat': 0, 'dog': np.log(3) / 3, 'eats': np.log(3 / 2) / 3,  'fish': 0, 'likes': 0},
        2: {'cat': np.log(3 / 2) / 3, 'dog': 0, 'eats': 0,  'fish': 0, 'likes': np.log(3) / 3}
    }

    for doc_id in expected_tf_idf_dict.keys():
        for term in expected_tf_idf_dict[doc_id].keys():
            assert abs(tf_idf_dict[doc_id][term] - expected_tf_idf_dict[doc_id][term]) < 1e-10

    # Unit test cho hàm cosine_similarity()
    vector_0 = list(tf_idf_dict[0].values())
    vector_1 = list(tf_idf_dict[1].values())
    cosine_similarity = vectorizer.cosine_similarity(vector_0, vector_1)
    assert abs(cosine_similarity - 0.2448297501) < 1e-10

    # Sau khi các unit test chạy thành công, in ra thông báo này. Ngược lại nếu có ít nhất 1 unit test thất bại sẽ trả về lỗi AssertionError
    print("All tests passed!")