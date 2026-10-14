# Lab 04 — Text Classification

Lab 04 áp dụng các cách biểu diễn văn bản vào bài toán phân loại cảm xúc review phim. Dataset IMDb có hai nhãn `positive` và `negative`. Tài liệu yêu cầu chi tiết nằm trong [`doc/W4.pdf`](doc/W4.pdf).

Notebook thực hiện khảo sát dữ liệu, so sánh TF-IDF với Average Word2Vec, khảo sát word order và n-gram, đánh giá trên test, confusion matrix, error analysis, inspect model weights và thử dự đoán review mới. Dữ liệu được chia thành train/validation/test theo tỷ lệ 60/20/20; TF-IDF và Word2Vec chỉ học từ dữ liệu huấn luyện để tránh data leakage.

## Cấu trúc thư mục

```text
Lab4/
├── README.md                    # Giới thiệu và hướng dẫn chạy
├── data/
│   └── IMDB Dataset.csv         # Dữ liệu: cột review và sentiment
├── doc/
│   └── W4.pdf                   # Đề bài Lab 04
├── src/
│   ├── classification.ipynb     # Các thí nghiệm, đánh giá và xuất kết quả
│   └── preprocessing.py         # Làm sạch, chia tập, tạo TF-IDF pipeline,
│                                # hỗ trợ Average Word2Vec
├── calculations.pdf             # Phần bài tập tính toán
├── prediction.md                # Dự đoán trước thí nghiệm
├── results.csv                  # Kết quả Experiment 2–5 và Evaluation
└── error_analysis.md            # Chưa có nội dung phân tích lỗi
```

`results.csv` hiện chứa các giá trị từ output đã lưu trong notebook. Chạy lại cell **Lưu kết quả Experiment 2–5 và Evaluation** để xuất CSV với giá trị mới nhất của phiên chạy. `confusion_matrix.png` chưa được lưu ra tệp; `error_analysis.md` vẫn đang trống.

## Chuẩn bị môi trường

- Dùng **Python 3.13**. Notebook hiện ghi phiên bản 3.13.15; trước đây `gensim` gặp lỗi với kernel Python 3.14.5.
- Cần các gói: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `gensim`, `ipykernel`.
- Trong workspace này có sẵn môi trường Python 3.13 tại `C:\NLP\Lab3\.venv\Scripts\python.exe`. Có thể chọn interpreter này làm kernel trong VS Code/Jupyter.

Nếu kernel được chọn thiếu thư viện, chạy lệnh sau trong PowerShell từ `C:\NLP\Lab4` rồi khởi động lại kernel:

```powershell
& '..\Lab3\.venv\Scripts\python.exe' -m pip install numpy pandas matplotlib seaborn scikit-learn gensim ipykernel
```

Kiểm tra kernel trong một cell notebook:

```python
import sys
print(sys.executable)
print(sys.version)
```

## Chạy `classification.ipynb`

1. Mở [`src/classification.ipynb`](src/classification.ipynb) trong VS Code hoặc Jupyter và chọn kernel Python 3.13 có đủ các gói trên.
2. Đặt **working directory** là `Lab4/src`. Notebook dùng đường dẫn tương đối `../data/IMDB Dataset.csv` và import `preprocessing` từ cùng thư mục. Nếu dùng Jupyter từ terminal, hãy mở terminal tại `Lab4/src` trước khi mở notebook.
3. Chạy các cell theo thứ tự từ trên xuống (`Run All`). Sau khi sửa `preprocessing.py`, hãy restart kernel và chạy lại từ đầu để nạp phiên bản mới.

Các bước notebook hiện thực:

1. **Experiment 1 — Dataset inspection:** đọc CSV; thống kê số mẫu, nhãn, missing/blank, duplicate và độ dài review; loại review trùng chính xác; vẽ phân phối độ dài và phân phối class.
2. **Data Leakage:** giải thích vì sao phải chia dữ liệu trước khi fit TF-IDF.
3. **Experiment 2 — Baseline:** `split_dataset(data)` chia 60/20/20 với `random_state=2026` và `stratify`; pipeline fit TF-IDF và Logistic Regression trên train, dự đoán validation, rồi hiển thị training time, vocabulary size, accuracy, precision, recall và F1.
4. **Experiment 3–5:** so sánh Average Word2Vec với TF-IDF, thử hai câu khác nhau bởi từ `not`, rồi so unigram với unigram + bigram trên validation.
5. **Evaluation:** fit mô hình TF-IDF unigram + bigram trên train + validation, tính metrics trên test. Cell tiếp theo ghi `../results.csv` từ các biến kết quả của Experiment 2–5 và Evaluation.
6. **Phân tích và Application:** vẽ confusion matrix, chọn ví dụ đúng/sai, xem trọng số feature và dự đoán review mới.

Các model chỉ tồn tại trong phiên chạy notebook; hiện chưa có bước lưu model ra tệp. Không dùng test để chọn model hoặc điều chỉnh tham số: test dành cho đánh giá cuối sau khi đã chọn cấu hình trên validation.

## Các hàm trong `preprocessing.py`

| Hàm | Vai trò |
| --- | --- |
| `clean_review(text)` | Giải mã HTML, bỏ HTML tag, chuẩn hóa khoảng trắng và chữ thường. |
| `tokenize_review(text)` | Tách token để dùng với Word2Vec. |
| `split_dataset(data)` | Chia train/validation/test theo tỷ lệ mặc định 60/20/20. |
| `build_tfidf_lr_pipeline(...)` | Tạo pipeline TF-IDF + Logistic Regression; TF-IDF được fit khi gọi `fit()` trên train. |
| `get_vocabulary_size(model)` | Lấy kích thước vocabulary của pipeline đã fit. |
| `build_average_word2vec_features(texts, model)` | Lấy trung bình vector từ để tạo vector cho mỗi review. |