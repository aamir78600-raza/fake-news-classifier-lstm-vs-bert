# fake-news-classifier-lstm-vs-bert
# 📰 Fake News Detection: LSTM vs BERT

A text classification project that detects whether a news article is **real or fake** using two deep learning approaches: a lightweight **LSTM** baseline and a fine-tuned **BERT** transformer. Both models ship with an interactive **Gradio** web demo.

## 📌 Overview

| | LSTM Model | BERT Model |
|---|---|---|
| Framework | TensorFlow / Keras | PyTorch / Hugging Face Transformers |
| Architecture | Embedding → LSTM(64) → Dropout → Dense | `bert-base-uncased` fine-tuned for sequence classification |
| Input length | 200 tokens (vocab 5,000) | 256 tokens |
| Epochs | 5 | 2 |
| Parameters | ~0.69M | ~110M |
| Test accuracy | **96.66%** | **99.99%** |

## 📂 Dataset

- ~45,000 news articles labelled as real (Reuters-style reporting) or fake
- Split: 80% training / 20% testing (`random_state=42`, 8,984 test samples)
- Only the `text` column is used; labels: `0 = Fake`, `1 = Real`

> The dataset is not included in this repository. Place `Fake_news.csv` and `True_News.csv` in the working directory before running the notebooks.

## 🧠 Approach

### 1. LSTM Baseline
- Text cleaning: lowercasing, removing URLs, HTML tags, brackets and punctuation
- Keras `Tokenizer` (5,000-word vocabulary) with padding to 200 tokens
- Embedding (128-d) → LSTM (64 units) → Dropout (0.5) → Sigmoid output
- Adam optimizer, binary cross-entropy, batch size 64

### 2. BERT Fine-Tuning
- `bert-base-uncased` tokenizer with truncation and padding (max length 256)
- `BertForSequenceClassification` fine-tuned with the Hugging Face `Trainer`
- 2 epochs, batch size 8

## 📊 Results

**LSTM:** test accuracy 96.66% (validation accuracy ~96–97.6% across epochs).

**BERT:** test accuracy 99.99%

```
Confusion Matrix        Predicted Fake   Predicted Real
True Fake                    4660              0
True Real                       1           4323
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Fake | 1.00 | 1.00 | 1.00 |
| Real | 1.00 | 1.00 | 1.00 |

> **Note:** Near-perfect scores on this type of dataset can partly come from source-specific cues (for example, publisher tags or writing style) rather than true fact-checking. Evaluate on data from other sources before drawing real-world conclusions.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/<your-username>/fake-news-detection-lstm-bert.git
cd fake-news-detection-lstm-bert
pip install pandas numpy scikit-learn tensorflow transformers datasets torch gradio matplotlib seaborn
```

### Run
1. Add `Fake_news.csv` and `True_News.csv` to the project directory (or Colab `/content/`).
2. Open a notebook from the `notebooks/` folder in Jupyter or Google Colab:
   - `News_Detector_LSTM_Model.ipynb`
   - `News_Detection_Using_BERT_Model.ipynb`
3. Run all cells. The last cell launches a Gradio app where you can paste an article and get a prediction with a confidence score.

## 🖥️ Demo

Paste any news article text into the Gradio interface and the model returns **Real** or **Fake** along with a confidence percentage.

## 🔮 Future Work

- Test on multiple, independent datasets to measure generalisation
- Compare with other transformers (RoBERTa, DistilBERT)
- Add headline + body analysis and explainability (LIME/SHAP)
- Deploy the app permanently on Hugging Face Spaces

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Gradio](https://www.gradio.app/)

## 📄 License

MIT License
