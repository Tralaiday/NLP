# LAB03 — Word Representations and Embeddings

## Cấu trúc thư mục

Các thư mục `data/` và `doc/` không được liệt kê dưới đây vì chỉ chứa dữ
liệu đầu vào và tài liệu bài lab.

```text
Lab3/
├── src/
│   ├── co_occurrence.py
│   └── word_embedding.ipynb
├── calculations.pdf
├── error_analysis.md
├── prediction.md
├── README.md
├── reflection.md
└── results.csv
```

### Vai trò của các file chính

- `src/co_occurrence.py`: cài đặt `CoOccurrenceModel` cho word-context
  representation và co-occurrence matrix của Experiment 1. File cũng có
  một đoạn test nhỏ trong khối `if __name__ == '__main__':`.
- `src/word_embedding.ipynb`: đọc và tiền xử lý corpus, huấn luyện Word2Vec,
  thực hiện các Experiment 2–4, evaluation và semantic search.
- `results.csv`: các kết quả thực nghiệm được xuất từ notebook.
- `prediction.md`: dự đoán trước khi chạy các experiment.
- `error_analysis.md`: phân tích các kết quả đúng, sai hoặc bất ngờ.
- `reflection.md`: phần reflection về static embedding và contextual
  embedding.
- `calculations.pdf`: phần tính toán lý thuyết.


## Chuẩn bị môi trường

Project sử dụng virtual environment `.venv` với Python 3.13 và các package
chính như `gensim`, `numpy`, `scipy`, `pandas` và `ipykernel`.

Trong PowerShell, chạy từ thư mục `Lab3`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu môi trường chưa có package, có thể cài bằng:

```powershell
uv pip install --python .\.venv\Scripts\python.exe gensim pandas ipykernel
```

## Chạy test `co_occurrence.py`

Đoạn test mẫu nằm ở cuối file `src/co_occurrence.py`. Có thể chạy từ thư
mục `Lab3` bằng:

```powershell
.\.venv\Scripts\python.exe .\src\co_occurrence.py
```

Hoặc chạy từ thư mục `src` sau khi kích hoạt môi trường:

```powershell
Set-Location .\src
python .\co_occurrence.py
```

Test mẫu sẽ:

1. Tạo một corpus nhỏ gồm các câu về `cat`, `dog`, `fish` và `milk`.
2. Xây dựng vocabulary và co-occurrence matrix với `k = 1`.
3. In vocabulary.
4. In ma trận.
5. In ba từ gần `eats` nhất bằng `most_similar('Eats', 3)`.

## Chạy `word_embedding.ipynb`

Mở file:

```text
src/word_embedding.ipynb
```

Trong VS Code:

1. Mở notebook.
2. Chọn kernel `Python (Lab3)` hoặc interpreter
   `.\.venv\Scripts\python.exe`.
3. Chạy các cell theo thứ tự từ trên xuống dưới.

Notebook sẽ lần lượt:

- chạy Experiment 1 với `CoOccurrenceModel`;
- đọc và tiền xử lý corpus;
- huấn luyện Word2Vec;
- chạy Experiment 3 về context window;
- chạy Experiment 4 về embedding dimension;
- đánh giá word similarity và word analogy;
- chạy downstream task và semantic search;
- lưu kết quả vào `results.csv`.

Có thể chạy notebook từ terminal bằng Jupyter nếu đã cài Jupyter:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook .\src\word_embedding.ipynb
```

Do notebook huấn luyện Word2Vec trên corpus lớn với nhiều cấu hình, một số
cell có thể mất nhiều thời gian. Không nên chạy lặp lại các cell huấn luyện
nếu chưa cần thiết.

Sau khi chạy xong các cell tạo DataFrame, chạy cell **Lưu kết quả** ở cuối
notebook để cập nhật:

```text
results.csv
```