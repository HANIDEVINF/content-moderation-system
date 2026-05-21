from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "public" / "model"
REPORT_DIR = ROOT / "reports"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MAX_FEATURES = 1600
SEED = 42


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def download_dataset() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "sms_spam_collection.zip"
    raw_path = DATA_DIR / "SMSSpamCollection"
    if not raw_path.exists():
        print("Downloading SMS Spam Collection dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(DATA_DIR)
    return raw_path


def load_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        label, text = line.split("\t", 1)
        texts.append(text)
        labels.append(1 if label == "spam" else 0)
    return texts, np.array(labels, dtype=np.float32)


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return {word: index for index, (word, _) in enumerate(counts.most_common(MAX_FEATURES))}


def vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = tokenize(text)
        if not tokens:
            continue
        for token in tokens:
            index = vocab.get(token)
            if index is not None:
                matrix[row, index] += 1.0
        matrix[row] = np.log1p(matrix[row])
    return matrix


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    texts, labels = load_dataset(download_dataset())
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    vocab = build_vocab(train_texts)
    x_train = vectorize(train_texts, vocab)
    x_val = vectorize(val_texts, vocab)
    x_test = vectorize(test_texts, vocab)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(vocab),), name="bag_of_words"),
            tf.keras.layers.Dense(96, activation="relu", name="dense_features"),
            tf.keras.layers.Dropout(0.35, name="dropout"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_reasoning"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=14,
        batch_size=64,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "dataset_url": DATA_URL,
        "train_size": len(train_texts),
        "validation_size": len(val_texts),
        "test_size": len(test_texts),
        "vocabulary_size": len(vocab),
        "architecture": ["Dense(96, relu)", "Dropout(0.35)", "Dense(32, relu)", "Dense(1, sigmoid)"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "epochs_ran": len(history.history["loss"]),
    }

    weights = model.get_weights()
    export = {
        "model_type": "keras_dense_bow_binary_classifier",
        "positive_label": "spam_or_unsafe",
        "negative_label": "ham_or_safe",
        "thresholds": {"allow": 0.35, "review": 0.65},
        "vocabulary": vocab,
        "weights": {
            "dense1_kernel": weights[0].tolist(),
            "dense1_bias": weights[1].tolist(),
            "dense2_kernel": weights[2].tolist(),
            "dense2_bias": weights[3].tolist(),
            "out_kernel": weights[4].tolist(),
            "out_bias": weights[5].tolist(),
        },
        "metrics": metrics,
    }

    model.save(MODEL_DIR / "keras_sms_moderation.keras")
    (MODEL_DIR / "moderation_model.json").write_text(json.dumps(export), encoding="utf-8")
    (REPORT_DIR / "keras_moderation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["ham_safe", "spam_unsafe"]),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
