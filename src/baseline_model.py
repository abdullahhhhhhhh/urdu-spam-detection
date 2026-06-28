"""
Baseline Urdu SMS Spam Detection Models
Implements: Naive Bayes, SVM (TF-IDF features), and Bidirectional LSTM
"""
import sys
import os
import json
import re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.sample_data import get_texts_and_labels, get_dataset_stats

try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[ERROR] scikit-learn not found. Run: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. LSTM model will be skipped.")


# ─────────────────────────────────────────────
# Urdu text preprocessing
# ─────────────────────────────────────────────

def preprocess_urdu(text):
    """Remove URLs, numbers, punctuation; normalize whitespace."""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[!.,،؟?:;()\[\]{}\-_/\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_char_vocab(texts):
    """Build character-level vocabulary for LSTM."""
    chars = set()
    for t in texts:
        chars.update(t)
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


def encode_text(text, vocab, max_len=100):
    """Encode text as list of char indices, padded/truncated to max_len."""
    encoded = [vocab.get(c, vocab['<UNK>']) for c in text]
    if len(encoded) < max_len:
        encoded += [vocab['<PAD>']] * (max_len - len(encoded))
    return encoded[:max_len]


def compute_metrics(y_true, y_pred, model_name):
    """Print and return classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# Classical ML: Naive Bayes and SVM
# ─────────────────────────────────────────────

def run_naive_bayes(X_train, X_test, y_train, y_test):
    """TF-IDF + Multinomial Naive Bayes."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return compute_metrics(y_test, y_pred, "Naive Bayes (TF-IDF char 3-gram)"), pipeline


def run_svm(X_train, X_test, y_train, y_test):
    """TF-IDF + Support Vector Machine."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return compute_metrics(y_test, y_pred, "SVM (TF-IDF char 3-gram + RBF kernel)"), pipeline


# ─────────────────────────────────────────────
# Deep Learning: Bidirectional LSTM
# ─────────────────────────────────────────────

class UrduSMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.encoded = [torch.tensor(encode_text(t, vocab, max_len), dtype=torch.long)
                        for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for Urdu SMS spam classification."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional → hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out)).squeeze(1)


def run_lstm(texts_train, texts_test, y_train, y_test, epochs=50):
    """Train and evaluate bidirectional LSTM."""
    if not TORCH_AVAILABLE:
        print("[SKIP] PyTorch not available — LSTM skipped.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}, None

    preprocessed_train = [preprocess_urdu(t) for t in texts_train]
    preprocessed_test = [preprocess_urdu(t) for t in texts_test]

    vocab = build_char_vocab(preprocessed_train + preprocessed_test)

    train_ds = UrduSMSDataset(preprocessed_train, y_train, vocab)
    test_ds = UrduSMSDataset(preprocessed_test, y_test, vocab)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
        logits = model(x_test)
        y_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

    return compute_metrics(y_test, y_pred, "Bidirectional LSTM (char-level)"), model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Urdu SMS Spam Detection — Baseline Models")
    print("=" * 60)

    print("\n--- Dataset Statistics ---")
    get_dataset_stats()

    texts, labels = get_texts_and_labels()
    texts = [preprocess_urdu(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    results = {}

    if SKLEARN_AVAILABLE:
        nb_metrics, _ = run_naive_bayes(X_train, X_test, y_train, y_test)
        results["naive_bayes"] = nb_metrics

        svm_metrics, _ = run_svm(X_train, X_test, y_train, y_test)
        results["svm"] = svm_metrics

    lstm_metrics, _ = run_lstm(X_train, X_test, y_train, y_test)
    results["lstm"] = lstm_metrics

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[Saved] results/baseline_results.json")

    return results


if __name__ == "__main__":
    main()
