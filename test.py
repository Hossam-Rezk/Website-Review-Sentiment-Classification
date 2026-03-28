
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


MODEL_PATHS = {
    'cnn': {
        'model': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\cnn_model.keras",
        'tokenizer': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\tokenizer.pkl",
        'label_encoder': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\label_encoder.pkl"
    },
    'transformer': {
        'model': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\best_transformer_v4_by_loss.keras",
        'tokenizer': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\tokenizer.pkl",
        'label_encoder': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\label_encoder.pkl"
    },
    'bilstm': {
        'model': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\bilstm_best.pt",
        'metadata': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\bilstm_metadata.pt"
    },
    'han': {
        'checkpoint': r"C:\Users\HOSSAM\Downloads\NN_project\nn-26-review-sentiment-classification\han_full_checkpoint.pt"
    }
}


CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "it's": "it is", "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will"
}

SAFE_STOPWORDS = {
    "the", "a", "an", "and", "is", "are", "was", "were", "to", "of", "in",
    "for", "on", "at", "by", "with", "from", "it", "this", "that", "these",
    "those", "as", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may", "might"
}


def clean_text_keras(text):
    """Clean text for Keras models (CNN/Transformer)"""
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_pytorch(text):
    """Clean text for PyTorch models (BiLSTM/HAN)"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    
    # Expand contractions
    for c, e in CONTRACTIONS.items(): 
        text = text.replace(c, e)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in SAFE_STOPWORDS]
    
    return " ".join(tokens)


def preprocess_text_han(text):
    """Preprocessing for HAN model"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'\d+', ' <NUM> ', text)
    text = re.sub(r'([?.!,ØŒØŸ])', r' \1 ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF<>ØŸ!.ØŒ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_sentences(text):
    """Split text into sentences for HAN"""
    return [s.strip() for s in re.split(r'(?<=[\.\!\?\ØŸ])\s+', text) if s.strip()]


class BiLSTMAttention(nn.Module):
    """BiLSTM with Attention mechanism"""
    def __init__(self, vocab_size, num_classes, d_model=256, hidden_dim=256, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.attention_v = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_u = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        u = torch.tanh(self.attention_v(output))
        a = self.attention_u(u).squeeze(2)
        mask = (x != 0)
        a = a.masked_fill(~mask, -1e9)
        weights = F.softmax(a, dim=1).unsqueeze(2)
        context = torch.sum(output * weights, dim=1)
        return self.classifier(context)


class AttentionLayer(nn.Module):
    """Attention layer for HAN"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_linear = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.context_vector = nn.Linear(2 * hidden_dim, 1, bias=False)

    def forward(self, rnn_output, mask):
        u_it = torch.tanh(self.attn_linear(rnn_output))
        scores = self.context_vector(u_it).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context_vector = (attention_weights * rnn_output).sum(dim=1)
        return context_vector


class HierarchicalAttentionNetwork(nn.Module):
    """HAN: Hierarchical Attention Network"""
    def __init__(self, vocab_size, num_classes, embed_dim=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attn = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.sent_rnn = nn.GRU(2 * hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attn = AttentionLayer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, max_sents, max_words = x.size()
        x_flat = x.view(batch_size * max_sents, max_words)
        mask_word = (x_flat != 0)
        embeddings = self.dropout(self.embedding(x_flat))
        word_out, _ = self.word_rnn(embeddings)
        sent_vectors = self.word_attn(word_out, mask_word)
        sent_vectors = sent_vectors.view(batch_size, max_sents, -1)
        mask_sent = (x.sum(dim=2) > 0)
        sent_out, _ = self.sent_rnn(sent_vectors)
        doc_vector = self.sent_attn(sent_out, mask_sent)
        return self.classifier(doc_vector)




def load_cnn_model(device='cpu'):
    """Load CNN model"""
    print("Loading CNN model...")
    
    model_path = MODEL_PATHS['cnn']['model']
    tokenizer_path = MODEL_PATHS['cnn']['tokenizer']
    label_encoder_path = MODEL_PATHS['cnn']['label_encoder']
    
    model = keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'max_len': 150  # Default from your notebook
    }


def load_transformer_model(device='cpu'):
    """Load Transformer model"""
    print("Loading Transformer model...")
    
    model_path = MODEL_PATHS['transformer']['model']
    tokenizer_path = MODEL_PATHS['transformer']['tokenizer']
    label_encoder_path = MODEL_PATHS['transformer']['label_encoder']
    
    model = keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'max_len': 150
    }


def load_bilstm_model(device='cpu'):
    """Load BiLSTM model"""
    print("Loading BiLSTM model...")
    
    model_path = MODEL_PATHS['bilstm']['model']
    metadata_path = MODEL_PATHS['bilstm']['metadata']
    
    meta = torch.load(metadata_path, map_location=device)
    stoi = meta['stoi']
    label_map = meta['label_map']
    max_len = meta['max_len']
    idx2label = {v: k for k, v in label_map.items()}
    
    model = BiLSTMAttention(
        vocab_size=len(stoi),
        num_classes=len(label_map),
        d_model=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return {
        'model': model,
        'stoi': stoi,
        'idx2label': idx2label,
        'max_len': max_len
    }


def load_han_model(device='cpu'):
    """Load HAN model"""
    print("Loading HAN model...")
    
    checkpoint_path = MODEL_PATHS['han']['checkpoint']
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    word_to_index = checkpoint['word_to_index']
    label_map = checkpoint['label_map']
    max_sentences = checkpoint['max_sentences']
    max_words = checkpoint['max_words']
    idx2label = {v: k for k, v in label_map.items()}
    
    model = HierarchicalAttentionNetwork(len(word_to_index), num_classes=len(label_map))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'word_to_index': word_to_index,
        'idx2label': idx2label,
        'max_sentences': max_sentences,
        'max_words': max_words
    }



def predict_cnn(test_df, model_dict):
    """Make predictions using CNN"""
    print("Preprocessing text for CNN...")
    X_test = test_df['text'].apply(clean_text_keras)
    
    X_test_seq = model_dict['tokenizer'].texts_to_sequences(X_test)
    X_test_pad = pad_sequences(
        X_test_seq, 
        maxlen=model_dict['max_len'], 
        padding='post', 
        truncating='post'
    )
    
    print("Making predictions...")
    predictions = model_dict['model'].predict(X_test_pad, verbose=1)
    pred_classes = predictions.argmax(axis=1)
    pred_labels = model_dict['label_encoder'].inverse_transform(pred_classes)
    
    return pred_labels


def predict_transformer(test_df, model_dict):
    """Make predictions using Transformer"""
    print("Preprocessing text for Transformer...")
    X_test = test_df['text'].apply(clean_text_keras)
    
    X_test_seq = model_dict['tokenizer'].texts_to_sequences(X_test)
    X_test_pad = pad_sequences(
        X_test_seq, 
        maxlen=model_dict['max_len'], 
        padding='post', 
        truncating='post'
    )
    
    print("Making predictions...")
    predictions = model_dict['model'].predict(X_test_pad, verbose=1)
    pred_classes = predictions.argmax(axis=1)
    pred_labels = model_dict['label_encoder'].inverse_transform(pred_classes)
    
    return pred_labels


def text_to_tensor_bilstm(texts, stoi, max_len=100):
    """Convert text to tensor for BiLSTM"""
    data = np.zeros((len(texts), max_len), dtype=np.int32)
    unk_idx = stoi.get('<UNK>', 1)
    
    for i, text in enumerate(texts):
        tokens = clean_text_pytorch(text).split()[:max_len]
        data[i, :len(tokens)] = [stoi.get(t, unk_idx) for t in tokens]
    
    return torch.tensor(data, dtype=torch.long)


def predict_bilstm(test_df, model_dict, device='cpu'):
    """Make predictions using BiLSTM"""
    print("Preprocessing text for BiLSTM...")
    
    X_test = text_to_tensor_bilstm(
        test_df['text'].tolist(), 
        model_dict['stoi'], 
        model_dict['max_len']
    ).to(device)
    
    print("Making predictions...")
    all_preds = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            preds = model_dict['model'](batch).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
    
    pred_labels = [model_dict['idx2label'][p] for p in all_preds]
    return pred_labels


def text_to_tensor_han(texts, word_to_index, max_sentences=15, max_words=30):
    """Convert text to tensor for HAN"""
    num_samples = len(texts)
    tensor = np.zeros((num_samples, max_sentences, max_words), dtype=np.int32)
    unknown_idx = word_to_index.get('<UNK>', 1)
    
    for i, text in enumerate(texts):
        sentences = get_sentences(preprocess_text_han(text))
        for j, sent in enumerate(sentences[:max_sentences]):
            words = sent.split()[:max_words]
            token_ids = [word_to_index.get(w, unknown_idx) for w in words]
            tensor[i, j, :len(token_ids)] = token_ids
    
    return torch.tensor(tensor, dtype=torch.long)


def predict_han(test_df, model_dict, device='cpu'):
    """Make predictions using HAN"""
    print("Preprocessing text for HAN...")
    
    X_test = text_to_tensor_han(
        test_df['text'].tolist(),
        model_dict['word_to_index'],
        model_dict['max_sentences'],
        model_dict['max_words']
    ).to(device)
    
    print("Making predictions...")
    all_preds = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            logits = model_dict['model'](batch)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
    
    pred_labels = [model_dict['idx2label'][p] for p in all_preds]
    return pred_labels



def main():
    parser = argparse.ArgumentParser(description='Test trained models on new data')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['cnn', 'transformer', 'bilstm', 'han'],
                       help='Model to use for prediction')
    parser.add_argument('--test_file', type=str, default='test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save predictions (default: predictions_<model>.csv)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for PyTorch models')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model.upper()}")
    print(f"Test file: {args.test_file}")
    print("="*50)
    
    print("\nLoading test data...")
    test_df = pd.read_csv(args.test_file)
    test_df['text'] = test_df['text'].fillna("").astype(str)
    print(f"Loaded {len(test_df)} samples")
    
    # Keep true labels if exist
    has_true = 'review' in test_df.columns
    if has_true:
        test_df.rename(columns={'review': 'review_true'}, inplace=True)
    
    # Load model and make predictions
    if args.model == 'cnn':
        model_dict = load_cnn_model()
        predictions = predict_cnn(test_df.copy(), model_dict)
    elif args.model == 'transformer':
        model_dict = load_transformer_model()
        predictions = predict_transformer(test_df.copy(), model_dict)
    elif args.model == 'bilstm':
        model_dict = load_bilstm_model(device)
        predictions = predict_bilstm(test_df.copy(), model_dict, device)
    elif args.model == 'han':
        model_dict = load_han_model(device)
        predictions = predict_han(test_df.copy(), model_dict, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Save predictions
    output_file = args.output_file or f'predictions_{args.model}.csv'
    test_df['review'] = predictions
    test_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Predictions saved to: {output_file}")
    print(f"{'='*50}")
    
    # Sample predictions
    print("\nSample predictions:")
    cols_to_show = ['id']
    if has_true:
        cols_to_show.append('review_true')
    cols_to_show.append('review')
    print(test_df[cols_to_show].head(10))
    
    # Distribution
    print("\nPrediction distribution:")
    print(test_df['review'].value_counts())
    
    # Accuracy
    if has_true:
        y_true = test_df['review_true'].astype(str)
        y_pred = test_df['review'].astype(str)
        acc = (y_true == y_pred).mean()
        print(f"\nAccuracy on provided labels: {acc:.4f}")


# ============================================================================
# GUI MODE
# ============================================================================

def run_gui():
    """Simple Tkinter GUI for unified_test."""
    root = tk.Tk()
    root.title("Unified Model Tester")

    selected_model = tk.StringVar(value="cnn")
    selected_file = tk.StringVar(value="")
    status_text = tk.StringVar(value="Select a CSV file and a model, then click Run.")

    def browse_file():
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            selected_file.set(path)
            status_text.set(f"Selected file: {os.path.basename(path)}")

    def run_inference():
        csv_path = selected_file.get().strip()
        model_name = selected_model.get()

        if not csv_path:
            messagebox.showerror("Error", "Please choose a CSV file first.")
            return

        if not os.path.exists(csv_path):
            messagebox.showerror("Error", f"File not found:\n{csv_path}")
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[GUI] Using device: {device}")

            test_df = pd.read_csv(csv_path)
            if 'text' not in test_df.columns:
                messagebox.showerror("Error", "CSV must contain a 'text' column.")
                return
            test_df['text'] = test_df['text'].fillna("").astype(str)

            has_true = 'review' in test_df.columns
            if has_true:
                test_df.rename(columns={'review': 'review_true'}, inplace=True)

            status_text.set(f"Loading {model_name.upper()} model... please wait")

            if model_name == 'cnn':
                model_dict = load_cnn_model()
                preds = predict_cnn(test_df.copy(), model_dict)
            elif model_name == 'transformer':
                model_dict = load_transformer_model()
                preds = predict_transformer(test_df.copy(), model_dict)
            elif model_name == 'bilstm':
                model_dict = load_bilstm_model(device)
                preds = predict_bilstm(test_df.copy(), model_dict, device)
            elif model_name == 'han':
                model_dict = load_han_model(device)
                preds = predict_han(test_df.copy(), model_dict, device)
            else:
                messagebox.showerror("Error", f"Unknown model: {model_name}")
                return

            test_df['review'] = preds

            accuracy_str = "N/A (no true labels in CSV)"
            if has_true:
                y_true = test_df['review_true'].astype(str)
                y_pred = test_df['review'].astype(str)
                acc = (y_true == y_pred).mean()
                accuracy_str = f"{acc:.4f}"

            base, ext = os.path.splitext(csv_path)
            out_path = f"{base}_pred_{model_name}.csv"
            test_df.to_csv(out_path, index=False)

            status_text.set(
                f"Done. Saved predictions to:\n{out_path}\n"
                f"Accuracy: {accuracy_str}"
            )
            messagebox.showinfo(
                "Success",
                f"Predictions saved to:\n{out_path}\n\nAccuracy: {accuracy_str}"
            )

        except Exception as e:
            status_text.set("Error during prediction. Check console for details.")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            raise

    frame = ttk.Frame(root, padding=20)
    frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Unified Model Tester", font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=3, pady=(0, 15)
    )

    ttk.Label(frame, text="Model:").grid(row=1, column=0, sticky="w", pady=5)
    model_combo = ttk.Combobox(
        frame,
        textvariable=selected_model,
        values=["cnn", "transformer", "bilstm", "han"],
        state="readonly",
        width=15
    )
    model_combo.grid(row=1, column=1, sticky="w", pady=5)

    ttk.Label(frame, text="CSV File:").grid(row=2, column=0, sticky="w", pady=5)
    file_entry = ttk.Entry(frame, textvariable=selected_file, width=50)
    file_entry.grid(row=2, column=1, sticky="w", pady=5)
    ttk.Button(frame, text="Browse...", command=browse_file).grid(
        row=2, column=2, padx=5, pady=5
    )

    run_button = ttk.Button(frame, text="Run Prediction", command=run_inference)
    run_button.grid(row=3, column=0, columnspan=3, pady=15)

    status_label = ttk.Label(frame, textvariable=status_text, wraplength=500, justify="left")
    status_label.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")

    root.mainloop()




if __name__ == "__main__":
    # If called with arguments → CLI mode
    # If no arguments → open GUI
    if len(sys.argv) > 1:
        main()
    else:
        run_gui()
