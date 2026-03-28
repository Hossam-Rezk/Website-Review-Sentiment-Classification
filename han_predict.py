# han_predict.py
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from HAN import HierarchicalAttentionNetwork, text_to_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

checkpoint = torch.load('han_full_checkpoint.pt', map_location=device)
word_to_index = checkpoint['word_to_index']
label_map = checkpoint['label_map']
max_sentences = checkpoint['max_sentences']
max_words = checkpoint['max_words']

index_to_label = {v: k for k, v in label_map.items()}
sorted_labels = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]


model = HierarchicalAttentionNetwork(len(word_to_index), num_classes=len(label_map))
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()


print("\n" + "="*30)
print("RUNNING VALIDATION CHECK")
print("="*30)

# Load original training data
df_train = pd.read_csv("train.csv").dropna(subset=['text', 'review'])
y_all_indices = df_train['review'].map(label_map).values

_, X_val_txt, _, y_val_indices = train_test_split(
    df_train['text'].astype(str).tolist(),
    y_all_indices,
    test_size=0.15,
    stratify=y_all_indices,
    random_state=42
)

print(f"Processing {len(X_val_txt)} validation samples...")
X_val_tensor = text_to_tensor(X_val_txt, word_to_index, max_sentences, max_words).to(device)

val_preds = []
batch_size = 64

with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
        batch = X_val_tensor[i : i+batch_size]
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()
        val_preds.extend(preds)

acc = accuracy_score(y_val_indices, val_preds)
print(f"\nValidation Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_val_indices, val_preds, target_names=sorted_labels))
print("="*30 + "\n")

print("Loading test.csv...")
df_test = pd.read_csv("test.csv")
df_test['text'] = df_test['text'].fillna("").astype(str)

X_test = text_to_tensor(df_test['text'].tolist(), word_to_index, max_sentences, max_words).to(device)

print(f"Predicting on {len(X_test)} test samples...")
all_preds = []

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i+batch_size]
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)


print("Saving results...")
df_test['review'] = [index_to_label[p] for p in all_preds]
df_test[['id', 'review']].to_csv("test2_predictions_han.csv", index=False)
print("Done! Saved 'test_predictions_han.csv'")
print(df_test.head())