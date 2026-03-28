# Website Review Sentiment Classification

Multi-class text classification of website user reviews into five sentiment categories using four deep learning architectures.

---

## Problem

Given a user review as free text, classify it into one of five sentiment levels:

| Label | Meaning |
|-------|---------|
| Excellent | Highest positive sentiment |
| Very Good | Strong positive sentiment |
| Good | Moderate positive sentiment |
| Bad | Negative sentiment |
| Very Bad | Strongly negative sentiment |

---

## Dataset

- **Train:** 7,000 labeled reviews
- **Test:** 3,000 unlabeled reviews (predictions submitted as CSV)
- **Class distribution:** Imbalanced — Very Good (2,469) and Excellent (2,335) dominate; Very Bad (524) is the minority class
- **Preprocessing:** Lowercasing, regex cleaning, contraction expansion, synonym-based augmentation to balance classes to 2,469 samples each

---

## Models Implemented

### 1. TextCNN (`Model.ipynb`)
Convolutional neural network for text classification using Keras/TensorFlow.

```
Embedding → Conv1D (128) → MaxPool → Conv1D (64) → GlobalMaxPool → Dense → Dropout → Output
```
- Vocab: 20,000 tokens, sequence length: 150
- Baseline comparison: TF-IDF + Logistic Regression (77.6% accuracy)

### 2. Custom Transformer (`Model.ipynb`)
Transformer encoder with multi-head self-attention built from scratch in Keras.

```
Embedding + Positional Encoding → TransformerBlock × N → GlobalAvgPool → Dense → Output
```

### 3. BiLSTM with Attention (`bilstm_train.py` / `bilstm_predict.py`)
Bidirectional LSTM with a custom Bahdanau-style attention mechanism in PyTorch.

```
Embedding → BiLSTM (256 hidden, 2 layers) → Attention Pooling → Dense → Output
```
- Vocab: 25,000 tokens, sequence length: 100
- Class-weighted loss to handle imbalance
- Gradient clipping at 1.0

### 4. Hierarchical Attention Network — HAN (`HAN.py`)
Document-level model that processes text at two levels: word-level and sentence-level.

```
Word Embeddings → Word BiGRU → Word Attention → Sentence vectors
Sentence vectors → Sentence BiGRU → Sentence Attention → Document vector → Classifier
```
- Captures document structure better than flat sequence models
- WeightedRandomSampler for handling class imbalance
- Label smoothing (0.2) for regularization

---

## Project Structure

```
├── Model.ipynb                          # TextCNN + Custom Transformer (Keras)
├── bilstm_train.py                      # BiLSTM training script (PyTorch)
├── bilstm_predict.py                    # BiLSTM inference + validation
├── HAN.py                               # Hierarchical Attention Network (PyTorch)
├── test.py                              # General test/evaluation utilities
├── train.csv                            # Training data
├── test.csv                             # Test data (no labels)
├── sample_submission.csv                # Submission format reference
├── test_predictions_cnn.csv             # TextCNN predictions
├── test_predictions_custom_transformer.csv  # Transformer predictions
├── test_predictions_han.csv             # HAN predictions
└── README.md
```

---

## How to Run

### TextCNN and Custom Transformer

```bash
# Open Model.ipynb in Jupyter
# Ensure train.csv and test.csv are in the same directory
# Run all cells in order
jupyter notebook Model.ipynb
```

### BiLSTM with Attention

```bash
# Install dependencies
pip install torch pandas scikit-learn numpy

# Train
python bilstm_train.py
# Outputs: bilstm_best.pt, bilstm_metadata.pt

# Generate predictions
python bilstm_predict.py
# Outputs: submission_bilstm.csv
```

### Hierarchical Attention Network

```bash
# Train
python HAN.py
# Outputs: han_best_weights.pt, han_full_checkpoint.pt
```

---

## Key Technical Decisions

**Class imbalance** — addressed three ways depending on the model:
- Synonym augmentation to oversample minority classes to equal size (TextCNN/Transformer)
- Class-weighted CrossEntropyLoss (BiLSTM)
- WeightedRandomSampler (HAN)

**Why four models?** — Each architecture captures text differently. CNNs capture local n-gram patterns. LSTMs capture sequential dependencies. Transformers capture global context via attention. HAN explicitly models document hierarchy. Comparing them across the same task reveals which inductive bias suits review sentiment best.

**Attention in BiLSTM** — rather than taking the final hidden state, the attention layer learns which words contribute most to the sentiment signal. This is particularly useful for reviews where the key sentiment words may appear anywhere in the text.

---

## Dependencies

```
# Keras/TensorFlow models
tensorflow>=2.12
scikit-learn
pandas
numpy
nltk

# PyTorch models
torch>=2.0
pandas
scikit-learn
numpy
```

---

## Author

**Hossam Rezk** — Ain Shams University, Computer Science, Class of 2026
