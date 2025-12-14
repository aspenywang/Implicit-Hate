# CMPU 366 — Project: Implicit Hate Reasoning
# ===========================================

import os
import gc
import random
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Optional: LIME (if installed)
try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("[Warning] lime not installed; explanations will be skipped.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------
# Data paths and labels
# -----------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "implicit-hate-corpus"

STAGE1_PATH = DATA_DIR / "implicit_hate_v1_stg1_posts.tsv"
STAGE2_PATH = DATA_DIR / "implicit_hate_v1_stg2_posts.tsv"

LABEL2ID = {
    "explicit_hate": 0,
    "implicit_hate": 1,
    "not_hate": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------
# Data loading
# ---------------

def load_stage1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[["post", "class"]].rename(columns={"post": "text", "class": "label"})
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def load_stage2_irony(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["implicit_class"] == "irony"].copy()
    df = df.rename(columns={"post": "text"})
    df["label"] = "implicit_hate"
    df["label_id"] = LABEL2ID["implicit_hate"]
    return df


# -----------------------
# TF-IDF Baseline
# -----------------------

def run_logreg_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n=== Logistic Regression baseline ===")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label_id"].values

    X_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label_id"].values

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        multi_class="multinomial",
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)

    print(
        classification_report(
            y_val,
            val_pred,
            target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))],
        )
    )

    acc = accuracy_score(y_val, val_pred)
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"Baseline accuracy: {acc:.3f}, macro-F1: {macro_f1:.3f}")

    return vectorizer, clf


# ---------------------------
# Transformer fine-tuning
# ---------------------------

class HateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_transformer(
    model_name,
    train_df,
    val_df,
    out_dir,
    num_epochs=3,
    demo=False,
):

    print(f"\n=== Training: {model_name} (demo={demo}) ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = HateDataset(train_df, tokenizer)
    val_dataset = HateDataset(val_df, tokenizer)

    if demo:
        epochs = 1
        train_bs = 8
        eval_bs = 16
    else:
        epochs = num_epochs
        train_bs = 16
        eval_bs = 32

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=2e-5,
        logging_steps=50,
        no_cuda=True,   # Force CPU for stability on M2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"{model_name} eval:", metrics)

    return model, tokenizer, trainer


# ------------------------------
# Stage 2 Irony diagnostic
# ------------------------------

def predict_with_trainer(trainer, df, tokenizer, max_length=128):
    dataset = HateDataset(df, tokenizer, max_length)
    outputs = trainer.predict(dataset)
    logits = outputs.predictions
    preds = logits.argmax(axis=-1)
    return preds, logits


def analyze_irony_subset(irony_df, trained_models):
    print("\n=== Irony diagnostic ===")

    for name, bundle in trained_models.items():
        print(f"\n[Model: {name}]")

        preds, _ = predict_with_trainer(
            bundle["trainer"],
            irony_df,
            bundle["tokenizer"],
        )

        true = irony_df["label_id"].values
        acc = accuracy_score(true, preds)
        macro_f1 = f1_score(true, preds, average="macro")

        print(f"Accuracy: {acc:.3f}, macro-F1: {macro_f1:.3f}")

        fn = np.where(preds != LABEL2ID["implicit_hate"])[0]
        print(f"False negatives: {len(fn)} / {len(irony_df)}")

<<<<<<< HEAD:train.py
        class_names_ordered = [ID2LABEL[i] for i in range(len(LABEL2ID))]

        #Run Lime
        for i in fn[:1]:
            text_sample = irony_df.iloc[i]["text"]
            predicted_label = ID2LABEL[int(preds[i])]
=======
        for i in fn[:5]:
            print("\n--- Example ---")
            print("Text:", irony_df.iloc[i]["text"])
            print("Pred:", ID2LABEL[int(preds[i])])
>>>>>>> parent of dbf26e2 (try lime):project_implicit_hate.py


# ---------------------
# Optional: LIME demo
# ---------------------

def lime_explain_example(text, model, tokenizer, class_names, num_features=10):
    if not HAS_LIME:
        print("LIME not installed.")
        return

    explainer = LimeTextExplainer(class_names=class_names)

    def predict(text_list):
        outputs = []
        model.eval()
        with torch.no_grad():
            for t in text_list:
                enc = tokenizer(
                    t,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                ).to(DEVICE)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                outputs.append(probs[0])
        return np.vstack(outputs)

    exp = explainer.explain_instance(text, predict, num_features=num_features)

    print("\n[LIME explanation]")
    for word, weight in exp.as_list():
        print(f"{word:>12}: {weight:+.3f}")



# -----------------------
# 5. Main Execution
# -----------------------
if __name__ == "__main__":
    # --- COMMAND LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Implicit Hate Speech Experiment Runner")

    # Flag to enable Demo Mode (Fast debug)
    parser.add_argument("--demo", action="store_true", help="Run in fast DEMO mode (200 samples, 1 model, 1 epoch).")

    # Flag to disable saving (Optional, useful for quick tests)
    parser.add_argument("--no_save", action="store_true", help="Disable model saving to disk.")

    args = parser.parse_args()

    # Set Configuration based on arguments
    DEMO_MODE = args.demo
    SAVE_MODELS = not args.no_save

    print(f"{'='*40}")
    print(f"MODE: {'DEMO (Fast Debug)' if DEMO_MODE else 'FULL EXPERIMENT'}")
    print(f"SAVING MODELS: {SAVE_MODELS}")
    print(f"{'='*40}")

    # 1. Load and Encode Data
    try:
        stage1_df_raw = load_stage1(STAGE1_PATH)
        stage1_df = encode_labels(stage1_df_raw)
    except NameError:
        print("Error: Helper functions (load_stage1, encode_labels) or paths (STAGE1_PATH) are not defined.")
        sys.exit(1)

    # 2. Prepare Splits & Settings based on Mode
    if DEMO_MODE:
        print("Stage 1 label counts (Total):")
        print(stage1_df["label"].value_counts())

        # Sample down for speed
        working_df = stage1_df.sample(n=200, random_state=SEED)
        train_df, val_df = train_test_split(
            working_df, test_size=0.2, random_state=SEED, stratify=working_df["label_id"]
        )

        print("\nDemo sizes:", len(train_df), "train,", len(val_df), "val")

        # Run Baseline (Only in demo for quick sanity check)
        tfidf_demo, logreg_demo = run_logreg_baseline(train_df, val_df)

        # Config for Demo
        target_models = ["bert-base-uncased"]
        num_epochs = 1

    else:
        # Full Dataset
        print("Full dataset size:", len(stage1_df))
        train_df, val_df = train_test_split(
            stage1_df, test_size=0.2, random_state=SEED, stratify=stage1_df["label_id"]
        )

        # Config for Full Run
        target_models = [
            "bert-base-uncased",
            "roberta-base",
            "xlnet-base-cased"
        ]
        num_epochs = 3 # Standard fine-tuning epochs

    # 3. Training Loop
    trained_models = {}

    for model_name in target_models:
        print(f"\n\n{'='*40}")
        print(f"STARTING PIPELINE: {model_name}")
        print(f"{'='*40}")

        # Memory Cleanup before starting a new model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Train
        model, tokenizer, trainer = train_transformer(
            model_name,
            train_df,
            val_df,
            out_dir=f"./output_{model_name}",
            num_epochs=num_epochs,
            demo=DEMO_MODE
        )

        # Store in dictionary for analysis
        trained_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "trainer": trainer
        }

        # Save model
        if SAVE_MODELS:
            save_path = f"./output/models/saved_{model_name}"
            print(f"Saving model to {save_path}...")
            # Ensure directory exists to prevent errors
            os.makedirs(save_path, exist_ok=True)
            trainer.save_model(save_path)
            tokenizer.save_pretrained(save_path)

    # 4. Final Analysis on Irony Subset
    print(f"\n\n{'='*40}")
    print("FINAL COMPARATIVE EVALUATION: IRONY SUBSET")
    print(f"{'='*40}")

    irony_df = load_stage2_irony(STAGE2_PATH)
    print("Irony subset size:", len(irony_df))

<<<<<<< HEAD:train.py
    analyze_irony_subset(irony_df, trained_models)

    print("\n Execution Complete.")
=======
    analyze_irony_subset(irony_df, trained_models)
>>>>>>> parent of dbf26e2 (try lime):project_implicit_hate.py
