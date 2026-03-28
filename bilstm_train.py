import re
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in SAFE_STOPWORDS]
    return " ".join(tokens)


def build_vocab(texts, max_vocab=25000):
    print("Building Vocabulary...")
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text).split())
    common_words = [w for w, c in counter.most_common(max_vocab) if c >= 2]
    itos = ['<PAD>', '<UNK>'] + common_words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def text_to_tensor(texts, stoi, max_len=100):
    data = np.zeros((len(texts), max_len), dtype=np.int32)
    unk_idx = stoi.get('<UNK>', 1)
    for i, text in enumerate(texts):
        tokens = clean_text(text).split()[:max_len]
        data[i, :len(tokens)] = [stoi.get(t, unk_idx) for t in tokens]
    return torch.tensor(data, dtype=torch.long)



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


        output, (h_n, c_n) = self.lstm(emb)


        u = torch.tanh(self.attention_v(output))
        a = self.attention_u(u).squeeze(2)

        mask = (x != 0)
        a = a.masked_fill(~mask, -1e9)

        weights = F.softmax(a, dim=1).unsqueeze(2)

        context = torch.sum(output * weights, dim=1)

        return self.classifier(context)



def train_model(train_dl, val_dl, model, device, epochs=25, save_name='bilstm_best.pt', class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    print(f"\n--- Training {save_name} ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_v, y_v in val_dl:
                x_v, y_v = x_v.to(device), y_v.to(device)
                preds = model(x_v).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_v.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_acc)

        print(f"Ep {epoch:02d} | Loss: {total_loss / len(train_dl):.4f} | Val Acc: {val_acc:.4f}", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(" | [Saved Best]")
        else:
            print("")

    return best_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv("train.csv").dropna(subset=['text', 'review']).reset_index(drop=True)
    labels = sorted(df['review'].unique())
    label_map = {l: i for i, l in enumerate(labels)}
    print("Labels:", label_map)

    X_raw = df['text'].astype(str).tolist()
    y_raw = df['review'].map(label_map).values

    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.15, stratify=y_raw, random_state=42)

    stoi, itos = build_vocab(X_train, max_vocab=25000)
    MAX_LEN = 100

    X_train_t = text_to_tensor(X_train, stoi, MAX_LEN)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = text_to_tensor(X_val, stoi, MAX_LEN)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    counts = Counter(y_train)
    weights_list = [len(y_train) / (len(counts) * counts[i]) for i in range(len(counts))]
    class_weights = torch.tensor(weights_list, dtype=torch.float).to(device)
    print("Class Weights:", weights_list)

    train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    model = BiLSTMAttention(
        vocab_size=len(stoi),
        num_classes=len(label_map),
        d_model=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.4
    ).to(device)

    train_model(train_dl, val_dl, model, device, epochs=25, class_weights=class_weights)

    torch.save({'stoi': stoi, 'label_map': label_map, 'max_len': MAX_LEN}, 'bilstm_metadata.pt')
    print("Training Complete!")