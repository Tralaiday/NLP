import numpy as np

class CoOccurrenceModel:

    def __init__(self):
        self.vocabulary = set()             # Tập các từ vựng
        self.document_tokenized_list = []   # Danh sách token theo document
        self.vocab_size = 0                 # Tổng số từ vựng trong corpus
        self.matrix = None                  # Co-occurence matrix

        self.word_to_index = dict()         # Dict đánh thứ tự word trong document

    # Hàm dùng để fit model từ corpus(xây tập từ vựng và co-occurence matrix)
    def fit(self, corpus, k: int):
        if k <= 0:
            raise ValueError("k phải là số nguyên dương")
        
        self.vocabulary.clear()
        self.document_tokenized_list.clear()
        self.word_to_index.clear()

        self.__build_vocabulary(corpus)
        self.__build_cooccurence_matrix(k)

        return self

    # Hàm xây tập từ vựng
    def __build_vocabulary(self, corpus):
        for document in corpus:
            term_list = document.strip('.!?\n').lower().replace(',', ' ').split()
            self.document_tokenized_list.append(term_list)
            for term in term_list:
                self.vocabulary.add(term)

        self.vocabulary = sorted({
            term for term in self.vocabulary
        })
        self.vocab_size = len(self.vocabulary)

    # Hàm xây co-occurence matrix
    def __build_cooccurence_matrix(self, k: int):
        self.word_to_index = {
            word: index
            for index, word in enumerate(self.vocabulary)
        }

        self.matrix = np.zeros(
            (self.vocab_size, self.vocab_size), dtype = int
        )

        for document in self.document_tokenized_list:
            for i in range(len(document)):
        
                target_word = document[i]
        
                left = max(0, i - k)
                right = min(i + k + 1, len(document))
        
                for j in range(left, right):
                    if j == i:
                        continue
        
                    context_word = document[j]
                    self.matrix[
                        self.word_to_index[target_word],
                        self.word_to_index[context_word]
                    ] += 1

    # Hàm tính cosine similarity
    def __cosine_similarity(self, vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        
        dot_product = np.dot(vector1, vector2)
                
        vec1_norm = np.linalg.norm(vector1)
        vec2_norm = np.linalg.norm(vector2)
        
        if vec1_norm == 0 or vec2_norm == 0:
            return 0.0
        
        return dot_product / (vec1_norm * vec2_norm)

    # Hàm tìm ra top-k từ có similar cao nhất với target_word
    def most_similar(self, target_word: str, top_k: int):
        if top_k <= 0:
            raise ValueError('top_k phải là số nguyên dương')
        
        target_word = target_word.lower()
        if target_word not in self.vocabulary:
            raise ValueError('Từ không có trong văn bản')

        most_similar_words = dict()
        target_word_vector = self.matrix[self.word_to_index[target_word], :]

        for word in self.word_to_index.keys():
            if target_word == word:
                continue

            index = self.word_to_index[word]
            most_similar_words[word] = self.__cosine_similarity(
                target_word_vector, self.matrix[index, :]
            )

        return sorted(
            most_similar_words.items(),
            key = lambda item: item[1],
            reverse = True
        )[:top_k]

# Phần test chương trình trên
if __name__ == '__main__':
    raw_corpus = [
        'the cat eats fish.',
        'the dog eats fish!',
        'the cat likes milk',
        'the dog likes meat.'
    ]

    model = CoOccurrenceModel()

    model.fit(raw_corpus, 1)

    print(model.vocabulary)
    print(model.matrix)
    print(model.most_similar('Eats', 3))