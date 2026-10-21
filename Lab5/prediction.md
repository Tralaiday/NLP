# Prediction 1:

Prediction: RNN + Embedding có khả năng xử lý thứ tự từ trực tiếp hơn TF-IDF + Logistic Regression.

Reason: RNN đọc review lần lượt từng từ và cập nhật hidden state sau mỗi bước. Còn TF-IDF unigram trong bài này chỉ đếm mức độ quan trọng của từng từ, nên đổi chỗ các từ thì vector gần như vẫn vậy.

Confidence: 50%.

# Prediction 2:

Prediction: LSTM có lẽ sẽ xử lý dependency dài tốt hơn RNN thường.

Reason: Hidden state của RNN dễ bị các từ xuất hiện sau ghi đè. LSTM có cell state và các gate để giữ lại hoặc bỏ bớt thông tin, nên từ quan trọng ở đầu review có cơ hội ảnh hưởng tới dự đoán cuối hơn.

Confidence: 50%.

# Prediction 3:

Prediction: Khi tăng sequence length từ 50 lên 200, thời gian train RNN sẽ tăng rõ rệt.

Reason: Model phải đi qua nhiều timestep hơn cho mỗi review, nên chưa chắc thời gian sẽ tăng đúng 4 lần vì nhiều review ngắn hơn 200 từ và còn phụ thuộc cách padding, batch.

Confidence: 50%.

# Prediction 4:

Prediction: Cắt mọi review còn 30 tokens có thể làm kết quả phân loại giảm.

Reason: Nhiều review mở đầu chỉ kể nội dung phim, còn nhận xét thật nằm ở phía sau. Nếu phần sau có câu kiểu "but the ending was excellent" mà bị cắt mất, model sẽ thiếu thông tin để quyết định nhãn.

Confidence: 50%.

# Prediction 5:

Prediction: LSTM không chắc đạt accuracy cao hơn RNN trên mọi dataset.

Reason: Nếu review ngắn và từ mang cảm xúc nằm ngay gần cuối, RNN có thể đã đủ dùng. LSTM phức tạp hơn nên kết quả còn tùy dữ liệu, cách chia tập và siêu tham số; không thể chỉ nhìn tên model mà đoán chắc.

Confidence: 50%.
