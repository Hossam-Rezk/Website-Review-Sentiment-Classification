import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, hidden_dim=256, num_layers=2, dropout=0.3):
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


def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    for c, e in CONTRACTIONS.items(): text = text.replace(c, e)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in SAFE_STOPWORDS]
    return " ".join(tokens)


def text_to_tensor(texts, stoi, max_len=100):
    data = np.zeros((len(texts), max_len), dtype=np.int32)
    unk_idx = stoi.get('<UNK>', 1)
    for i, text in enumerate(texts):
        tokens = clean_text(text).split()[:max_len]
        data[i, :len(tokens)] = [stoi.get(t, unk_idx) for t in tokens]
    return torch.tensor(data, dtype=torch.long)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BiLSTM on {device}...")

    try:
        meta = torch.load('bilstm_metadata.pt', map_location=device)
        stoi = meta['stoi']
        label_map = meta['label_map']
        max_len = meta['max_len']
        idx2label = {v: k for k, v in label_map.items()}
    except Exception as e:
        print(f"Error loading metadata: {e}")
        exit()

    model = BiLSTMAttention(
        vocab_size=len(stoi),
        num_classes=len(label_map),
        d_model=256, hidden_dim=256, num_layers=2, dropout=0
    ).to(device)

    try:
        model.load_state_dict(torch.load('bilstm_best.pt', map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()

    print("\n--- Validation Check ---")
    try:
        df_train = pd.read_csv("train.csv").dropna(subset=['text', 'review'])
        y_all = df_train['review'].map(label_map).values

        _, X_val_txt, _, y_val = train_test_split(
            df_train['text'].astype(str).tolist(), y_all,
            test_size=0.15, stratify=y_all, random_state=42
        )

        X_val_t = text_to_tensor(X_val_txt, stoi, max_len).to(device)

        val_preds = []
        with torch.no_grad():
            for i in range(0, len(X_val_t), 64):
                batch = X_val_t[i:i + 64]
                preds = model(batch).argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)

        acc = accuracy_score(y_val, val_preds)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, val_preds,
                                    target_names=[k for k, v in sorted(label_map.items(), key=lambda x: x[1])]))

    except Exception as e:
        print(f"Error during validation check: {e}")

    print("\n--- Generating Test Predictions ---")
    try:
        df_test = pd.read_csv("test.csv")
        df_test['text'] = df_test['text'].fillna("").astype(str)

        X_test = text_to_tensor(df_test['text'].tolist(), stoi, max_len).to(device)
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_test), 64):
                batch = X_test[i:i + 64]
                preds = model(batch).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)

        df_test['review'] = [idx2label[p] for p in all_preds]
        df_test[['id', 'review']].to_csv("submission_bilstm.csv", index=False)
        print("Done! Saved 'submission_bilstm.csv'")
    except Exception as e:
        print(f"Error generating test predictions: {e}")