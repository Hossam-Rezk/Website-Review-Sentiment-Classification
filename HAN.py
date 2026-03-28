# han_train.py
import re
import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score



def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'\d+', ' <NUM> ', text)
    text = re.sub(r'([?.!,،؟])', r' \1 ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF<>؟!.،]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[\.\!\?\؟])\s+', text) if s.strip()]


def build_vocabulary(texts, max_vocab_size=15000):
    print("Building Vocabulary...")
    word_counts = Counter()
    for text in texts:
        for sentence in get_sentences(text):
            word_counts.update(sentence.split())

    common_words = [word for word, count in word_counts.most_common(max_vocab_size) if count >= 2]

    index_to_word = ['<PAD>', '<UNK>'] + common_words
    word_to_index = {word: i for i, word in enumerate(index_to_word)}

    return word_to_index


def text_to_tensor(texts, word_to_index, max_sentences=15, max_words=30):

    print(f"Converting text to tensor (Sentences={max_sentences}, Words={max_words})...")
    num_samples = len(texts)
    tensor = np.zeros((num_samples, max_sentences, max_words), dtype=np.int32)
    unknown_idx = word_to_index.get('<UNK>', 1)

    for i, text in enumerate(texts):
        sentences = get_sentences(preprocess_text(text))

        for j, sent in enumerate(sentences[:max_sentences]):
            words = sent.split()[:max_words]

            token_ids = [word_to_index.get(w, unknown_idx) for w in words]
            tensor[i, j, :len(token_ids)] = token_ids

    return torch.tensor(tensor, dtype=torch.long)




class AttentionLayer(nn.Module):
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


def train_model(train_loader, val_loader, model, device, epochs=15):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    best_f1 = 0.0

    print(f"\nStarting training on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                preds = model(x_val).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        scheduler.step(val_f1)
        avg_loss = total_loss / len(train_loader.dataset)

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'han_best_weights.pt')

    return model



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("train.csv").dropna().reset_index(drop=True)

    labels_unique = sorted(df['review'].unique())
    label_map = {label: i for i, label in enumerate(labels_unique)}
    print(f"Classes: {label_map}")

    X_raw = df['text'].astype(str).tolist()
    y_raw = df['review'].map(label_map).values

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.15, stratify=y_raw, random_state=42
    )

    word_to_index = build_vocabulary(X_train)
    max_sentences, max_words = 15, 30

    X_train_tensor = text_to_tensor(X_train, word_to_index, max_sentences, max_words).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val_tensor = text_to_tensor(X_val, word_to_index, max_sentences, max_words).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    model = HierarchicalAttentionNetwork(len(word_to_index), num_classes=len(label_map)).to(device)
    train_model(train_loader, val_loader, model, device, epochs=20)

    torch.save({
        'model_state': model.state_dict(),
        'word_to_index': word_to_index,
        'label_map': label_map,
        'max_sentences': max_sentences,
        'max_words': max_words
    }, 'han_full_checkpoint.pt')
    print("Training Complete. Saved 'han_full_checkpoint.pt'")