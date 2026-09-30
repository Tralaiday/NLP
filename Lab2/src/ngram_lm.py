from collections import Counter
import math

class NGramLanguageModel:

    def __init__(self, n, smoothing = False):
        """
        Hàm dựng khởi tạo mô hình ngôn ngữ n-gram.

        Input:
            n: Bậc của mô hình, chỉ nhận 1, 2 hoặc 3.
            smoothing: Áp dụng kỹ thuật smoothing nếu là True.

        Raises:
            ValueError: Nếu n không phải 1, 2 hoặc 3.
        """
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3")
        
        self.n = n
        self.smoothing = smoothing

        self.vocabulary = set()         # Lưu tập từ vựng trong corpus
        self.ngram_counts = Counter()   # Lưu số lần xuất hiện của từng n-gram.
        self.context_counts = Counter() # Lưu số lần xuất hiện của từng context
        self.total_tokens = 0           # Tổng số token xuất hiện trong corpus
        self.model_type = None          # Lưu loại mô hình (Unigram, Bigram, Trigram)

    def __build_vocabulary(self, corpus):
        """
        Xây dựng tập các từ vựng duy nhất xuất hiện trong toàn bộ corpus

        Input:
            corpus: Danh sách các câu, mỗi câu là danh sách token

        Output:
            Cập nhật từ điển
        """
        self.vocabulary = {
            word
            for sentence in corpus
            for word in sentence
        }

    def __count_ngrams(self, corpus):
        """
        Hàm đếm tổng số token, số lần xuất hiện của n-gram và context.

        Với unigram:
            - ngram là một token.
            - context được quy ước bằng None.

        Với bigram hoặc trigram:
            - ngram là tuple gồm n token.
            - context là n - 1 token đầu tiên.

        Input:
            corpus: Danh sách các câu, mỗi câu là danh sách token

        Output:
            Cập nhật self.total_tokens, self.ngram_counts, self.context_counts
        """
        for sentence in corpus:
            self.total_tokens += len(sentence)

            for index in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[index:index + self.n])

                if self.n == 1:
                    ngram = ngram[0]
                    context = None
                else:
                    context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def __train_unigram(self):
        """
        Đánh dấu mô hình hiện tại là unigram.
        """
        self.model_type = 'unigram'

    def __train_bigram(self):
        """
        Đánh dấu mô hình hiện tại là bigram.
        """
        self.model_type = 'bigram'

    def __train_trigram(self):
        """
        Đánh dấu mô hình hiện tại là trigram.
        """
        self.model_type = 'trigram'

    def fit(self, corpus):
        """
        Huấn luyện mô hình trên corpus

        Input:
            corpus: Danh sách các câu, mỗi câu là danh sách token

        Output:
            Trả về đối tượng mô hình sau khi huấn luyện
        """

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_tokens = 0

        self.__build_vocabulary(corpus)
        self.__count_ngrams(corpus)

        if self.n == 1:
            self.__train_unigram()
        elif self.n == 2:
            self.__train_bigram()
        else:
            self.__train_trigram()

        return self

    def probability(self, context, word):
        """
        Tính xác suất của một token dựa trên context

        Input:
            context: Danh sách các token trước word đang xét
            word: Token đang xét

        Output:
            Xác suất của word theo context
        """

        # Với trường hợp unigram: P(word) = Count(word) / Tổng số token
        if self.n == 1:
            ngram = word
            context_key = None
        # Với trường hợp bigram/trigram: P (word | context) = Count(context, word) / Count(context)
        else:
            context = tuple(context[-(self.n - 1):])    # Lấy ra n - 1 token cuối của context làm context
            ngram = context + (word,)
            context_key = context

        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context_key]

        # Công thức smoothing: P(word | context) = (Count(context, word) + 1) / (Count(context) + V)
        if self.smoothing:
            vocabulary_size = len(self.vocabulary)
            return (ngram_count + 1) / (context_count + vocabulary_size)

        # Nếu không smoothing và số lần context xuất hiện bằng 0, trả về xác suất bằng 0
        if context_count == 0:
            return 0.0

        return ngram_count / context_count

    def sentence_probability(self, sentence):
        """
        Tính xác suất của câu bằng tích xác suất có điều kiện của từng token trong câu

        Input:
            sentence: Danh sách token trong câu.

        Output:
            Xác suất của câu
        """
        # Công thức: P(w1, w2, ..., wK) = P(w1) P(w2 | w1) ... P(wK | context)
        probability = 1.0

        for index, word in enumerate(sentence):
            # Unigram không sử dụng context nên P(w1, w2, ..., wK) = P(w1) P(w2) ... P(wK)
            if self.n == 1:
                context = []
            else:
                context = sentence[
                    max(0, index - self.n + 1):index
                ]

            probability *= self.probability(context, word)

        return probability

    def sentence_log_probability(self, sentence):
        """
        Tính log-probability của một câu.

        Việc sử dụng log giúp tránh underflow khi câu dài hoặc khi xác suất của từng token rất nhỏ.

        Args:
            sentence: Danh sách token trong câu.

        Returns:
            Tổng log xác suất của câu.
            Trả về -inf nếu có token có xác suất bằng 0.
        """
        # Công thức: log(P(w1, ..., wK)) = log(P(w1)) + log(P(w2 | w1)) + ... + log(P(wK | context))
        log_probability = 0.0

        for index, word in enumerate(sentence):
            if self.n == 1:
                context = []
            else:
                context = sentence[
                    max(0, index - self.n + 1):index
                ]

            probability = self.probability(context, word)

            # Xử lý khi xác suất bằng 0
            if probability == 0:
                return float('-inf')

            log_probability += math.log(probability)

        return log_probability

    def next_word_distribution(self, context):
        """
        Tính phân phối xác suất của từ tiếp theo.

        Input:
            context: Các token gần nhất trước vị trí cần dự đoán.

        Output:
            Dictionary ánh xạ mỗi token trong vocabulary
            tới xác suất xuất hiện sau context.
        """
        # Distribution có thể lọc ra từ có xác suất cao nhất và sinh văn bản
        distribution = {}

        for word in self.vocabulary:
            distribution[word] = self.probability(context, word)

        return distribution