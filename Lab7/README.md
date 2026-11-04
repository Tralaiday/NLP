# LAB 07 — Transformer Architecture

## Run

Install PyTorch and run from the repository root:

```bash
python Lab7/src/run_lab7.py
```

The script uses `Lab7/data/IMDB Dataset.csv`. For a quick local check:

```bash
LAB7_MAX_SAMPLES=1000 LAB7_EPOCHS=1 python Lab7/src/run_lab7.py
```

The script compares LSTM and Transformer classification, reports precision,
recall and F1, and saves `learning_curves.png` and `confusion_matrix.png` when
matplotlib is available. The notebook contains the positional encoding,
multi-head, head-diversity and sequence-length experiments.
