# Prediction 1:

Prediction: TF-IDF + Logistic Regression có thể làm baseline tốt.

Reason: Trong Dataset IMDB, mô hình này có thể phân loại review dựa trên những từ khoa nổi bật, chẳng hạn như `"love"`, `"great"`.

Confidence: 70%.

# Prediction 2:

Prediction: Average Word2Vec chưa chắc tốt hơn so với TF-IDF.

Reason: Average Word2Vec có thể làm mất trật tự từ trong review, dẫn đến độ tương đồng giữa 2 câu bất kỳ có thể cao dù ngữ nghĩa của 2 câu hoàn toàn khác nhau.

Confidence: 50%.

# Prediction 3:

Prediction: Nếu tăng vocabulary size thì accuracy chưa chắc được cải thiện.

Reason: Trong quá trình train model có thể học các từ nhiễu hay cực kỳ hiếm, dẫn đến sai sót trong việc phân loại các mẫu văn bản mới.

Confidence: 50%.

# Prediction 4:

Prediction: Nếu bỏ stopword thì performance có thể giảm xuống.

Reason: Model sẽ bỏ qua các stopword để phân loại nhãn văn bản, chẳng hạn như `"not"`, `"no"`

Confidence: 50%