# Prediction 1:

Prediction: Key đó sẽ có attention weight lớn hơn.

Reason: dot product càng lớn thì key đó càng phù hợp với query, từ đó model sẽ chú ý hơn đến token tương ứng.

Confidence: 50%

# Prediction 2:

Prediction: Các attention weights sẽ bằng nhau.

Reason: Vì không có score nào nổi bật, model sẽ phân bổ sự chú ý đồng đều cho cả 4 token.

Confidence: 75%


# Prediction 3:

Prediction: Attention sẽ tập trung hoàn toàn vào token thứ ba

Reason: Score thứ ba vượt trội so với hai score còn lại, nên sau khi qua softmax, token thứ ba sẽ nhận attention weight gần bằng 1, còn hai token đầu có weight gần bằng 0.

Confidence: 80%

# Prediction 4:

Prediction: Chi phí tính attention sẽ tăng lên rất nhiều, khoảng 10.000 lần nếu xét số phép tính attention giữa các cặp token.

Reason: Mỗi token cần tính mức độ liên quan với tất cả token khác. Khi số token tăng, số lượng phép tính tăng theo bình phương độ dài sequence.

Confidence: 60%

# Prediction 5:

Prediction: Không

Reason: Mô hình xác định mức độ quan trọng dựa trên những đặc trưng và mối quan hệ đã học được từ dữ liệu, trong khi con người đánh giá dựa trên ngữ nghĩa và ngữ cảnh. Vì thế, token mà mô hình tập trung chú ý nhiều nhất chưa chắc là token mà con người cho là quan trọng nhất.

Confidence: 50%