# Prediction 1:

Prediction: Cặp từ (doctor - physician) gần nhau nhất

Reason: 2 từ này xuất hiện nhiều khi đặt trong ngữ cảnh tương tự nhau nên semantic similar giữa chúng có thể cao

Confidence: 70%

# Prediction 2:

Prediction: Có thay đổi

Reason: Khi context window tăng lên thì co-occurence sẽ tăng và mỗi từ sẽ đặt trong context rộng hơn

Confidence: 50%

# Prediction 3:

Prediction: Nếu embedding dimension tăng thì chất lượng model chưa chắc tăng

Reason: Model có thể học nhiễu trong corpus, dẫn đến overfitting.

Confidence: 50%

# Prediction 4:

Prediction: Cặp từ <"doctor", "physician"> chưa chắc gần nhau nếu corpus có gần 100 câu.

Reason: Mức độ similarity phụ thuộc vào context đang xét, nếu đặt trong cùng 1 context tương tự thì cặp từ trên có thể gần nhau.

Confidence: 50%

# Bài tập Prediction - CBOW và Skip-gram

- CBOW:

+ (cat) -> the
+ (the, eats) -> cat
+ (cat, fish) -> eats
+ (eats) -> fish

- Skip-gram:

+ the -> cat
+ cat -> the
+ cat -> eats
+ eats -> cat
+ eats -> fish
+ fish -> eats