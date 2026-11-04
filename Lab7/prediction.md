# Prediction 1:

Prediction: Một Transformer có thể xử lý các token trong sequence song song tốt hơn RNN.

Reason: RNN xử lý tuần tự vì hidden state tại một timestep phụ thuộc vào hidden state trước đó. Trong khi, Transformer sử dụng Self-Attention để tính toán mối quan hệ giữa các token thông qua các phép toán ma trận, nên việc xử lý song song sẽ tốt hơn.

Confidence: 75%

# Prediction 2:

Prediction: Nếu bỏ positional encoding, Transformer không thể phân biệt thứ tự token.

Reason: Self-Attention chủ yếu tính toán mức độ liên quan giữa các token thông qua Query và Key, chứ không tự chứa thông tin về vị trí. Do đó, nếu không bổ sung positional information, Transformer khó nhận biết thứ tự của các token.

Confidence: 80%

# Prediction 3:

Prediction: Nếu 2 câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information chưa chắc phân biệt được 2 câu này.

Reason: Hai câu chứa cùng các token thì có cùng tập token embeddings. Khi không có positional information, Self-Attention chỉ dựa trên nội dung các embeddings để tính toán, nên khó phân biệt hai câu dựa vào thứ tự. Đặc biệt, nếu sử dụng mean pooling, biểu diễn của hai câu có thể giống nhau.

Confidence: 50%

# Prediction 4:

Prediction: Hai attention heads không nhất thiết học cùng một attention pattern

Reason: Mỗi attention head có các ma trận projection Q, K, V riêng nên có thể học cách tập trung vào các token khác nhau. Vì vậy, các attention heads không nhất thiết tạo ra cùng một attention pattern.

Confidence: 75%

# Prediction 5:

Prediction: Tăng số attention heads chưa thể làm accuracy tăng

Reason: Nhiều attention heads có thể giúp mô hình học các mối quan hệ đa dạng hơn. Tuy nhiên, khi giữ cố định `d_model`, tăng số heads sẽ làm giảm số chiều của từng head. Ngoài ra, các heads có thể học thông tin trùng lặp, nên accuracy không chắc chắn tăng.

Confidence: 75%

# Prediction 6:

Prediction: Transformer chưa chắc nhanh hơn RNN trên mọi sequence length

Reason: Transformer có lợi thế xử lý song song, nhưng Self-Attention phải tính toán mối quan hệ giữa các cặp token, khiến chi phí tăng đáng kể khi sequence dài. Vì vậy, tốc độ còn phụ thuộc vào độ dài sequence, cấu hình mô hình và phần cứng sử dụng.

Confidence: 50%