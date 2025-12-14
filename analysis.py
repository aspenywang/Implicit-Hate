"""
Implicit Hate Speech Analysis Pipeline (Consolidated & CLI-Ready)
-----------------------------------------------------
Usage:
    python analysis.py --demo
    python analysis.py --data_dir ./my_data --model_dir ./my_models
    python analysis.py --lime_samples 1000
"""

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import warnings
import string
import nltk
from collections import Counter
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from safetensors.torch import load_file
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns


# --- 1. SETUP & UTILS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Labels (Constants)
LABEL2ID = {"explicit_hate": 0, "implicit_hate": 1, "not_hate": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# NLTK Setup
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(["cls", "sep", "pad", "unk", "sentence", "user", "rt", "http", "https"])

class HateDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_stage1_data(path):
    print(f"Loading Stage 1 data from {path}...")
    df = pd.read_csv(path, sep="\t", on_bad_lines='skip', engine='python')
    # Normalize column names
    if "post" in df.columns: df = df.rename(columns={"post": "text"})
    if "text" not in df.columns and "post" in df.columns: df = df.rename(columns={"post": "text"})

    valid_classes = ["explicit_hate", "implicit_hate", "not_hate"]
    if "class" in df.columns:
        df = df[df["class"].isin(valid_classes)].copy()

    df["label_id"] = df["class"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df.reset_index(drop=True)

def load_irony_data(path):
    print(f"Loading Irony data (Stage 2) from {path}...")
    df = pd.read_csv(path, sep="\t", on_bad_lines='skip', engine='python')
    if "implicit_class" in df.columns:
        df = df[df["implicit_class"] == "irony"].copy()
    if "post" in df.columns:
        df = df.rename(columns={"post": "text"})

    df["label_id"] = LABEL2ID["implicit_hate"]
    return df.reset_index(drop=True)

# --- 2. MODEL LOADING ---

def load_models(model_paths, device):
    models = {}
    print(f"\n{'='*20} LOADING MODELS {'='*20}")

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[WARNING] Path {path} not found. Skipping.")
            continue

        print(f"Loading {name} from {path}...")
        try:
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'r') as f: config_dict = json.load(f)
            config_dict["id2label"] = ID2LABEL
            config_dict["label2id"] = LABEL2ID
            config_dict["num_labels"] = 3

            if "model_type" in config_dict: model_type = config_dict.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_dict)

            model = AutoModelForSequenceClassification.from_config(config)
            safetensors_path = os.path.join(path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
            model.to(device)

            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
            except:
                print(f"   -> Local tokenizer missing. Downloading {name} from Hub...")
                tokenizer = AutoTokenizer.from_pretrained(name)

            trainer = Trainer(model=model, args=TrainingArguments(output_dir="/tmp/dumb", per_device_eval_batch_size=32, report_to="none"))
            models[name] = {"model": model, "tokenizer": tokenizer, "trainer": trainer}
            print(f"   -> [SUCCESS]")

        except Exception as e:
            print(f"[ERROR] Loading {name}: {e}")

    return models

# --- 3. PREDICTIONS ---

def generate_predictions(models, df):
    print(f"\n{'='*20} GENERATING PREDICTIONS {'='*20}")
    preds_dict = {}

    for name, bundle in models.items():
        print(f"Predicting with {name}...")
        ds = HateDataset(df, bundle["tokenizer"])
        raw_preds = bundle["trainer"].predict(ds)
        preds_dict[name] = raw_preds.predictions.argmax(axis=-1)

    print("[SUCCESS] Predictions complete.")
    return preds_dict

# --- 4. LIME HELPERS ---

def get_top_features_aggregate(texts, model, tokenizer, num_samples, device, max_examples):
    feature_counter = Counter()

    def predict_proba(text_list):
        model.eval()
        outputs = []
        with torch.no_grad():
            for t in text_list:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True,
                                   max_length=128).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                outputs.append(probs[0])
        return np.vstack(outputs)

    explainer = LimeTextExplainer(class_names=["explicit", "implicit", "not_hate"])

    limit = min(len(texts), max_examples)
    sample_texts = np.random.choice(texts, limit, replace=False) if limit < len(texts) else texts

    for text in tqdm(sample_texts, leave=False, desc="LIME Aggregate"):
        try:
            exp = explainer.explain_instance(text, predict_proba, num_features=5, num_samples=num_samples)
            pred_idx = np.argmax(exp.predict_proba)
            for word, score in exp.as_list(label=pred_idx):
                if score > 0:
                    feature_counter[word.lower()] += score
        except:
            continue
    return feature_counter

def get_lime_features_single(text, model, tokenizer, device, num_features=5, num_samples=500):
    def predict_proba(text_list):
        model.eval()
        outputs = []
        with torch.no_grad():
            for t in text_list:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                outputs.append(probs[0])
        return np.vstack(outputs)

    explainer = LimeTextExplainer(class_names=["explicit", "implicit", "not_hate"])
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=num_samples)
    return exp.as_list(), np.argmax(exp.predict_proba)

def precompute_lime_features(models, stage1_df, irony_df, lime_samples, max_examples, device):
    print(f"\n{'='*20} PRE-COMPUTING LIME FEATURES (RQ1-3) {'='*20}")
    cache = {name: {} for name in models.keys()}

    sources = {
        "Explicit": stage1_df[stage1_df["label_id"] == 0]["text"].tolist(),
        "Implicit": stage1_df[stage1_df["label_id"] == 1]["text"].tolist(),
        "Irony":    irony_df["text"].tolist()
    }

    for model_name, bundle in models.items():
        print(f"Processing {model_name}...")
        for category, texts in sources.items():
            if not texts: continue
            print(f"   -> Analyzing '{category}'...")
            counter = get_top_features_aggregate(texts, bundle["model"], bundle["tokenizer"],
                                                 num_samples=lime_samples, device=device, max_examples=max_examples)
            cache[model_name][category] = counter

    print("[SUCCESS] Pre-computation complete.")
    return cache

def save_lime_results(lime_cache, output_root):
    print(f"\n{'='*20} SAVING LIME RESULTS {'='*20}")

    output_dir = os.path.join(output_root, "lime_results")
    os.makedirs(output_dir, exist_ok=True)

    for model_name, categories in lime_cache.items():
        safe_name = model_name.split('/')[-1].replace('-', '_')
        for category, counter in categories.items():
            if not counter: continue

            df = pd.DataFrame(counter.most_common(), columns=["word", "score"])
            filename = f"lime_{safe_name}_{category.lower()}.csv"
            full_path = os.path.join(output_dir, filename)

            df.to_csv(full_path, index=False)
            print(f"   -> Saved {full_path}")

# --- 5. REPORTING & DIAGNOSTICS ---

def report_rq1(lime_cache):
    print(f"\n{'='*20} RQ1: Explicit vs Implicit Features {'='*20}")
    for model_name, categories in lime_cache.items():
        if "Explicit" not in categories or "Implicit" not in categories: continue
        print(f"\nModel: {model_name}")
        explicit_top = [w for w, s in categories["Explicit"].most_common(50) if w not in STOPWORDS][:7]
        implicit_top = [w for w, s in categories["Implicit"].most_common(50) if w not in STOPWORDS][:7]
        print(f"   Top Explicit: {explicit_top}")
        print(f"   Top Implicit: {implicit_top}")

def report_rq2(lime_cache):
    print(f"\n{'='*20} RQ2: General Implicit vs Irony {'='*20}")
    for model_name, categories in lime_cache.items():
        if "Implicit" not in categories or "Irony" not in categories: continue
        print(f"\nModel: {model_name}")
        gen_imp = [w for w, s in categories["Implicit"].most_common(50) if w not in STOPWORDS]
        irony_imp = [w for w, s in categories["Irony"].most_common(50) if w not in STOPWORDS]
        unique_irony = list(set(irony_imp) - set(gen_imp))[:7]
        print(f"   Unique Irony Drivers: {unique_irony}")

def report_rq3(lime_cache, plot_dir):
    print(f"\n{'='*20} RQ3: Interpretability Breakdown {'='*20}")
    results = []
    for model_name, categories in lime_cache.items():
        if "Irony" not in categories: continue
        counter = categories["Irony"]
        counts = {"Semantic": 0, "Syntactic": 0, "Noise": 0}
        total_analyzed = 0

        for word, score in counter.most_common(100):
            total_analyzed += 1
            if (word in string.punctuation) or (word.startswith("##")) or (len(word) < 2 and word not in ['a', 'i']):
                counts["Noise"] += 1
            elif word in STOPWORDS:
                counts["Syntactic"] += 1
            else:
                counts["Semantic"] += 1

        if total_analyzed > 0:
            results.append({
                "Model": model_name,
                "Semantic": (counts["Semantic"]/total_analyzed)*100,
                "Syntactic": (counts["Syntactic"]/total_analyzed)*100,
                "Noise": (counts["Noise"]/total_analyzed)*100
            })

    if results:
        df_res = pd.DataFrame(results)
        print("\nFeature Category Breakdown (% of Top 100):")
        print(df_res)

        # Plotting
        try:
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "rq3_breakdown.png")

            df_res.set_index("Model").plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
            plt.title("RQ3: Stability of Explanations")
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"   -> Saved plot to {plot_path}")
        except Exception as e:
            print(f"   [Warning] Could not plot: {e}")

def run_deep_diagnostics(models, irony_df, preds_dict, device, lime_samples, max_examples, plot_dir):
    print(f"\n{'='*20} DEEP DIAGNOSTICS & AGREEMENT ANALYSIS {'='*20}")

    short_names = [n.split('-')[0].upper() for n in models.keys()]
    model_keys = list(models.keys())
    error_sets = {}

    # 1. Calculate Errors and Unique Saves
    unique_saves = {}

    for i, name in enumerate(model_keys):
        # Error = Prediction is NOT 1 (Implicit Hate)
        errors = np.where(preds_dict[name] != 1)[0]
        error_sets[short_names[i]] = set(errors)

    if len(model_keys) >= 2:
        for current_model in model_keys:
            my_correct = set(np.where(preds_dict[current_model] == 1)[0])
            # Indices where others failed (prediction != 1)
            others_indices = [set(np.where(preds_dict[m] != 1)[0]) for m in model_keys if m != current_model]

            if others_indices:
                all_others_wrong = set.intersection(*others_indices)
                my_saves = my_correct.intersection(all_others_wrong)
                unique_saves[current_model] = list(my_saves)

    # 2. Venn Diagram
    if len(models) == 3:
        try:
            os.makedirs(plot_dir, exist_ok=True)
            venn_path = os.path.join(plot_dir, "venn_diagram.png")

            plt.figure(figsize=(8, 8))
            venn3([error_sets[n] for n in short_names], set_labels=short_names)
            plt.title("Overlap of False Negatives (Missed Irony)")
            plt.savefig(venn_path)
            print(f"   -> Saved Venn diagram to {venn_path}")
        except:
            pass

    # 3. Universal Failures Analysis
    if len(error_sets) > 0:
        all_fail_indices = list(set.intersection(*error_sets.values()))
        any_fail_indices = set.union(*error_sets.values())

        if len(any_fail_indices) > 0:
            pct_all_fail = len(all_fail_indices) / len(any_fail_indices) * 100
        else:
            pct_all_fail = 0.0

        print(f"\n[Systematic Failures]")
        print(f"Cases where ALL models failed: {len(all_fail_indices)}")
        print(f"Percentage of error pool: {pct_all_fail:.2f}%")

        if len(all_fail_indices) > 0:
            print(f"\n[ANALYSIS] Analyzing Universally Hard Cases...")
            sample_indices = all_fail_indices[:5]
            for i, idx in enumerate(sample_indices):
                text = irony_df.iloc[idx]["text"]
                print(f"\n--- Hard Case {i+1} ---")
                print(f"Text: \"{text}\"")
                print(f"Predictions: {[preds_dict[m][idx] for m in model_keys]} (All Wrong)")

                for name, bundle in models.items():
                    feats, _ = get_lime_features_single(text, bundle["model"], bundle["tokenizer"],
                                                        device, num_features=3, num_samples=lime_samples)
                    top_words = [f"{w} ({s:.2f})" for w, s in feats]
                    print(f"   {name.split('-')[0].upper()}: {', '.join(top_words)}")

    # 4. Divergence Analysis (Unique Saves)
    print(f"\n{'='*20} 4. DIVERGENCE LIME ANALYSIS {'='*20}")

    if len(unique_saves) > 0:
        # Find the model with the most unique saves
        best_saver_name = max(unique_saves, key=lambda k: len(unique_saves[k]))
        all_save_indices = unique_saves[best_saver_name]

        limit = min(len(all_save_indices), max_examples)
        save_indices = all_save_indices[:limit]

        if len(save_indices) > 0:
            print(f"Analyzing {len(save_indices)} divergence cases for {best_saver_name}...")
            bundle = models[best_saver_name]
            divergence_features = []

            for idx in tqdm(save_indices, desc="Analyzing Saves"):
                text = irony_df.iloc[idx]["text"]
                feats, pred_idx = get_lime_features_single(text, bundle["model"], bundle["tokenizer"],
                                                           device, num_features=5, num_samples=lime_samples)
                for word, score in feats:
                    if score > 0:
                        divergence_features.append(word)

            top_divergence_words = Counter(divergence_features).most_common(10)
            print(f"\n[Report Generation Data]")
            print(f"In {len(all_save_indices)} total cases, {best_saver_name} correctly identified the irony where the others failed.")
            print(f"LIME analysis suggests {best_saver_name} attended to:")
            print(f"{top_divergence_words}")

            if top_divergence_words:
                try:
                    os.makedirs(plot_dir, exist_ok=True)
                    div_plot_path = os.path.join(plot_dir, f"divergence_{best_saver_name}.png")

                    x, y = zip(*top_divergence_words)
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=list(y), y=list(x), palette="Greens_d")
                    plt.title(f"Unique Features Spotted by {best_saver_name}")
                    plt.xlabel("Frequency")
                    plt.tight_layout()
                    plt.savefig(div_plot_path)
                    print(f"   -> Saved Divergence plot to {div_plot_path}")
                except Exception as e:
                    print(f"   [Warning] Could not save divergence plot: {e}")
        else:
            print("No unique saves found (Models agreed on everything or failed together).")
    else:
        print("Skipping Divergence Analysis (Not enough models or no divergence found).")

    # 5. False Negatives (Positivity Trap)
    print(f"\n[Positivity Trap Analysis]")
    ground_truth = irony_df["label_id"].values
    fooling_words_map = {}

    for name, bundle in models.items():
        fn_indices = np.where((preds_dict[name] == 2) & (ground_truth == 1))[0]
        if len(fn_indices) == 0: continue

        limit = min(len(fn_indices), max_examples)
        target_indices = np.random.choice(fn_indices, limit, replace=False)
        fooling_words = []

        print(f"Analyzing {name} ({limit} failures)...")
        for idx in tqdm(target_indices, leave=False, desc="Positivity LIME"):
            text = irony_df.iloc[idx]["text"]
            feats, _ = get_lime_features_single(text, bundle["model"], bundle["tokenizer"],
                                                device, num_features=5, num_samples=lime_samples)
            for word, score in feats:
                if score > 0: fooling_words.append(word)

        fooling_words_map[name] = fooling_words

    # Plot Positivity
    if fooling_words_map:
        try:
            os.makedirs(plot_dir, exist_ok=True)
            positivity_path = os.path.join(plot_dir, "positivity_trap.png")

            fig, axes = plt.subplots(1, len(fooling_words_map), figsize=(18, 6))
            if len(fooling_words_map) == 1: axes = [axes]

            for i, (name, words) in enumerate(fooling_words_map.items()):
                counts = Counter(words).most_common(10)
                if not counts: continue
                x, y = zip(*counts)
                sns.barplot(x=list(y), y=list(x), ax=axes[i], palette="Reds_d")
                axes[i].set_title(f"{name.split('-')[0].upper()} Safe Signals")

            plt.tight_layout()
            plt.savefig(positivity_path)
            print(f"   -> Saved Positivity Trap plot to {positivity_path}")
        except Exception as e:
            print(f"   [Warning] Could not save positivity plot: {e}")

# --- 6. MAIN EXECUTION ---

def parse_args():
    parser = argparse.ArgumentParser(description="Implicit Hate Speech Analysis Pipeline")

    # Paths
    parser.add_argument("--data_dir", type=str, default="implicit-hate-corpus",
                        help="Root directory containing dataset TSV files")
    parser.add_argument("--model_dir", type=str, default="fine-tuned_models",
                        help="Root directory containing model subfolders")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save CSVs and plots")

    # Files
    parser.add_argument("--stage1_file", type=str, default="implicit_hate_v1_stg1_posts.tsv",
                        help="Filename for Stage 1 data")
    parser.add_argument("--stage2_file", type=str, default="implicit_hate_v1_stg2_posts.tsv",
                        help="Filename for Stage 2 data")

    # Mode and Parameters
    parser.add_argument("--demo", action="store_true", help="Run in fast demo mode (few samples)")
    parser.add_argument("--lime_samples", type=int, default=500, help="Number of LIME perturbation samples")
    parser.add_argument("--max_examples", type=int, default=797, help="Max examples to process for aggregates")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Config Logic
    if args.demo:
        print(f"\n{'!'*40}\n[INFO] DEMO MODE (5 samples)\n{'!'*40}\n")
        args.lime_samples = 5
        args.max_examples = 5
    else:
        print(f"\n{'='*40}\n[INFO] FULL ANALYSIS\n{'='*40}\n")

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Paths
    ROOT = Path(".").resolve()
    DATA_PATH = ROOT / args.data_dir
    STAGE1_FULL = DATA_PATH / args.stage1_file
    STAGE2_FULL = DATA_PATH / args.stage2_file

    # Ensure plot directory exists inside output_dir
    PLOT_DIR = os.path.join(args.output_dir, "plots")

    # Model Dictionary Construction
    # Assumes standard structure: model_dir/output_model_name/checkpoint
    model_definitions = {
        "bert-base-uncased": os.path.join(args.model_dir, "output_bert-base-uncased/checkpoint-1611"),
        "roberta-base": os.path.join(args.model_dir, "output_roberta-base/checkpoint-1611"),
        "xlnet-base-cased": os.path.join(args.model_dir, "output_xlnet-base-cased/checkpoint-1611")
    }

    # Execution
    models = load_models(model_definitions, DEVICE)

    if not models:
        print("[ERROR] No models loaded. Check paths provided via --model_dir")
        exit(1)

    irony_df = load_irony_data(STAGE2_FULL)
    stage1_df = load_stage1_data(STAGE1_FULL)

    predictions = generate_predictions(models, irony_df)

    lime_cache = precompute_lime_features(models, stage1_df, irony_df,
                                          lime_samples=args.lime_samples,
                                          max_examples=args.max_examples,
                                          device=DEVICE)

    save_lime_results(lime_cache, args.output_dir)
    report_rq1(lime_cache)
    report_rq2(lime_cache)
    report_rq3(lime_cache, plot_dir=PLOT_DIR)

    run_deep_diagnostics(models, irony_df, predictions,
                         device=DEVICE,
                         lime_samples=args.lime_samples,
                         max_examples=args.max_examples,
                         plot_dir=PLOT_DIR)

    print("\n[SUCCESS] Analysis pipeline complete.")