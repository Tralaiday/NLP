# LAB 05 — Bài tập về nhà: Sequence Models cho Sentiment Classification

> **Chủ đề:** Từ biểu diễn tĩnh đến mô hình tuần tự
> **Dữ liệu:** IMDb Movie Reviews (`data/IMDB Dataset.csv`)
> **Tiền đề:** LAB 04 — TF-IDF/Word2Vec và classification
> **Thời lượng khuyến nghị:** 6–8 giờ
> **Hình thức:** cá nhân

## 1. Câu hỏi trung tâm

> Khi giữ nguyên dữ liệu, cách chia tập và ngân sách huấn luyện, mô hình tuần tự
> (RNN/LSTM/GRU) sử dụng thông tin thứ tự tốt hơn baseline tĩnh đến mức nào,
> và chi phí phải trả là gì?

Toàn bộ bài làm phải quay lại câu hỏi này. Không cần triển khai quá nhiều thí nghiệm
nhỏ; ưu tiên một so sánh công bằng, có thể tái lập và được giải thích bằng cơ chế
model.

## 2. Kết quả cần đạt

Sau bài tập, sinh viên có thể:

1. Giải thích `hidden state`, recurrent computation và vấn đề long-range dependency.
2. Tự cài đặt và kiểm chứng một bước của `SimpleRNNCell`.
3. Xây dựng pipeline `text -> token IDs -> padding/truncation -> embedding -> classifier`.
4. Thiết kế thí nghiệm công bằng và phân tích accuracy, F1, chi phí tính toán và lỗi.

## 3. Quy ước thực nghiệm bắt buộc

Các quy ước này giúp kết quả giữa các bài có thể so sánh:

- Dùng đúng `data/IMDB Dataset.csv`; không tải thêm dữ liệu ngoài.
- Chia `train/validation/test` theo tỉ lệ `80/10/10`, cố định `random seed = 42`.
- Chỉ xây vocabulary trên tập train. Phải có token `<PAD>` và `<UNK>`.
- Báo cáo rõ tokenizer, vocabulary size, `max_length`, batch size, optimizer,
  learning rate và số epoch.
- Dùng cùng preprocessing, split, embedding dimension, batch size, optimizer,
  learning rate và số epoch khi so sánh các recurrent model.
- Chọn checkpoint tốt nhất theo validation F1; chỉ dùng test đúng một lần ở cuối.
- Không cần tự cài đặt LSTM/GRU hoặc toàn bộ training framework.

Nếu máy yếu, được giảm số epoch hoặc số mẫu nhưng phải ghi rõ thay đổi và dùng
đúng cùng cấu hình cho tất cả model.

## 4. Cấu trúc bài tập

### Phần A — Hiểu cơ chế (20 điểm)

Tạo `answers.md`, trả lời ngắn gọn nhưng có công thức hoặc ví dụ khi cần:

1. Tính một bước RNN với:
   - `x_t = [1, 0]^T`
   - `h_(t-1) = [0.5, 0.2]^T`
   - `W_xh = [[0.5, 0.2], [0.1, 0.4]]`
   - `W_hh = [[0.3, 0.1], [0.2, 0.5]]`
   - `b_h = [0, 0]^T`

   Trình bày `a_t` và `h_t = tanh(a_t)` đến ít nhất 4 chữ số thập phân.
2. Với hai câu `dog bites man` và `man bites dog`, giải thích vì sao
   Average Word2Vec có thể cho biểu diễn giống nhau nhưng RNN không nhất thiết như vậy.
3. Mô tả đường đi của thông tin trong câu
   `I did not like this movie`; giải thích vai trò của từ `not`.
4. Giải thích vanishing gradient bằng trực giác, sau đó phân biệt vai trò của
   forget gate trong LSTM với hidden state của RNN.

### Phần B — Xây dựng pipeline và kiểm chứng cell (25 điểm)

#### B1. Khảo sát dữ liệu

Trong notebook, báo cáo:

- số mẫu, số lớp và phân bố lớp;
- số review rỗng/trùng lặp;
- min, median, mean và max số token;
- histogram độ dài review;
- lý do chọn `max_length`.

#### B2. Pipeline dữ liệu

Viết các hàm hoặc class có trách nhiệm rõ ràng để:

```text
raw review
  -> tokenize
  -> vocabulary (train only)
  -> encode
  -> truncate/pad
  -> (input_ids, length, label)
```

`length` phải là độ dài thật trước padding để model không học từ `<PAD>`.

#### B3. `SimpleRNNCell`

Tạo `src/rnn_cell.py` với API tối thiểu:

```python
class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        ...

    def forward(self, x_t, h_prev):
        ...
```

Cell phải thực hiện:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

Kiểm chứng bằng input nhỏ có giá trị cố định và so sánh với
`torch.nn.RNNCell` khi dùng cùng weights. Sai số tối đa phải được báo cáo.

### Phần C — Thí nghiệm so sánh có kiểm soát (35 điểm)

Huấn luyện bốn model:

| Model | Input representation | Mục đích |
|---|---|---|
| TF-IDF + Logistic Regression | TF-IDF unigram/bigram | baseline tĩnh |
| Embedding + RNN | token sequence | recurrent baseline |
| Embedding + LSTM | token sequence | gated recurrence |
| Embedding + GRU | token sequence | gated recurrence gọn hơn |

Chỉ thay đổi recurrent architecture ở ba model neural. Bảng kết quả phải có:

| Model | Params | Best epoch | Train time | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | | | | | | | |
| RNN | | | | | | | |
| LSTM | | | | | | | |
| GRU | | | | | | | |

Ngoài bảng tổng hợp, bắt buộc có:

- train loss và validation loss theo epoch cho ba model neural;
- confusion matrix của model neural tốt nhất;
- một thí nghiệm phụ về `max_length`: chọn **hai** giá trị khác nhau (ví dụ 50 và
  200), giữ nguyên các yếu tố còn lại, và báo cáo `% tokens retained`, thời gian
  train, validation F1 và test F1.

Không được kết luận “model tốt hơn” chỉ từ accuracy. Phải giải thích trade-off giữa
context, F1 và chi phí huấn luyện.

### Phần D — Error analysis và kết luận (20 điểm)

Phân tích tối thiểu 6 review từ test set:

- 3 dự đoán đúng, trong đó có ít nhất một review có negation hoặc contrast;
- 3 dự đoán sai.

Với mỗi review, ghi text rút gọn, nhãn thật, nhãn dự đoán, độ tin cậy và nguyên nhân
khả dĩ. Nhóm nguyên nhân nên xem xét: negation, contrast, sarcasm, từ hiếm,
review dài, truncation, nhiễu hoặc ambiguity. Không được mặc định mọi lỗi là do
“RNN không hiểu context”.

Kết luận 1–2 trang phải trả lời trực tiếp:

1. Model nào phù hợp nhất với cấu hình và dữ liệu này? Bằng chứng là gì?
2. Sequence model giải quyết được hạn chế nào của TF-IDF/average embedding?
3. Kết quả nào cho thấy recurrent model vẫn có giới hạn?
4. Nếu làm tiếp LAB 06 về attention, vấn đề nào sẽ được giải quyết?

## 5. Cấu trúc nộp bài

```text
Lab5/
├── README.md
├── answers.md
├── report.md
├── data/
│   └── IMDB Dataset.csv
├── src/
│   ├── data_utils.py
│   ├── rnn_cell.py
│   ├── sequence_models.py
│   └── sequence_classification.ipynb
└── results/
    ├── metrics.csv
    ├── learning_curves.png
    ├── confusion_matrix.png
    └── error_analysis.csv
```

`report.md` phải chứa bảng kết quả, biểu đồ và kết luận. Notebook phải chạy được
từ đầu đến cuối sau khi cài dependencies; không nộp kết quả chỉ tồn tại trong
output của notebook mà không có file kết quả tương ứng.

## 6. Chấm điểm

| Thành phần | Điểm |
|---|---:|
| A. Hiểu cơ chế và tính toán | 20 |
| B1. Khảo sát dữ liệu và preprocessing | 10 |
| B2. `SimpleRNNCell` và kiểm chứng | 15 |
| C. Thiết kế, thực hiện và báo cáo thí nghiệm | 35 |
| D. Error analysis và kết luận | 15 |
| Tính tái lập, tổ chức code và trình bày | 5 |
| **Tổng** | **100** |

Trừ điểm nếu split bị rò rỉ, vocabulary được xây từ test, chọn model bằng test set,
hoặc so sánh model với các hyperparameter không công bằng.

## 7. Sử dụng AI

Được dùng AI cho boilerplate, debugging, giải thích API, vẽ biểu đồ và xử lý lỗi
runtime. Không được dùng AI để thay thế phần tính toán, prediction trước thí nghiệm,
giải thích kết quả, error analysis hoặc kết luận.

Nếu dùng AI, thêm cuối `report.md`:

```text
Tool:
Purpose:
Generated:
Modified:
Verified:
```

Sinh viên phải giải thích được code và kết quả trong phần kiểm tra cá nhân.