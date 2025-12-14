# Implicit Hate

This repository contains the code and data used for a CMPU366 course project on implicit hate speech classification and reasoning. The goal is to evaluate how different models handle high-level hate categories (Stage 1) and whether they generalize to context-dependent forms such as irony (Stage 2).

-----------------------------

This folder includes:

* `train.py`  
  The main script implementing:
  * Data loading for Stage 1 and Stage 2 posts  
  * Label encoding for `explicit_hate`, `implicit_hate`, and `not_hate`  
  * A small end-to-end demo pipeline  
  * TF-IDF + Logistic Regression baseline  
  * Transformer fine-tuning (BERT, RoBERTa, XLNet)  
  * Stage-2 irony diagnostic evaluation  
  * Optional LIME explanations

* `analysis.py`
 The post-training evaluation script implementing:

  * Loading of fine-tuned Transformer models (BERT, RoBERTa, XLNet)
  * Automated generation of predictions on the Stage 2 "Irony" diagnostic subset 
  * LIME-based feature interpretability analysis 
  * Deep diagnostic reporting, including Venn diagrams of error overlaps
  * False Positive analysis to identify misleading safe signals 
  * Command-line interface (CLI) for flexible path configuration and demo modes

* `implicit-hate-corpus/`  
  This directory contains the dataset used in the project:

  * `implicit_hate_v1_stg1_posts.tsv`  
    High-level annotations with:
    * `post`: the text of the tweet  
    * `class`: {`explicit_hate`, `implicit_hate`, `not_hate`}  

  * `implicit_hate_v1_stg2_posts.tsv`  
    Fine-grained implicit hate annotations with:
    * `post`  
    * `implicit_class`: {`white_grievance`, `incitement`, `inferiority`, `irony`, `stereotypical`, `threatening`, `other`}  
    * `extra_implicit_class`: same set as above or **None**  
    Only entries labeled `irony` are used for the diagnostic subset.
* `fine-tuned_models/`
  This directory contains the transformer models we fine-tuned:
  * Subdirectories for each model architecture (e.g., output_bert-base-uncased, output_roberta-base)
  * Saved model checkpoints (.safetensors or pytorch_model.bin)
  * Configuration files (config.json) and tokenizer data needed for loading the models in analysis.py

-----------------------------

## Setup & Installation

### 0. Clone the Repository
Start by cloning this repository to your local machine and navigating into the project directory:

```bash
git clone https://github.com/aspenywang/Implicit-Hate
cd implicit-hate-project. 
````

### 1. Requirements
Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`.

```bash
pip install -r requirements.txt
```

*Note: `matplotlib-venn` is optional but recommended for generating error overlap diagrams.*

### 2. Directory Setup
Ensure your data and models are organized. By default, the script expects:

* Data in: `./implicit-hate-corpus/`
* Models in: `./fine-tuned-models/`

If your paths differ, you can specify them via command-line arguments (see below).

---

## Usage
  You can execute the pipeline via the terminal. The script supports several flags to control execution mode, paths, and analysis fidelity.

### Quick Start (Demo Mode)
To run a fast sanity check (processing only 5 examples) to ensure the pipeline works:

```bash
python analysis.py --demo
```

### Full Analysis
To run the full analysis on the entire dataset with standard settings:

```bash
python analysis.py
```

### Custom Configurations
You can override default paths and settings using arguments:

| Argument | Description | Default                |
| --- | --- |------------------------|
| `--data_dir` | Path to the dataset folder | `implicit-hate-corpus` |
| `--model_dir` | Path to the saved models folder | `fine-tuned-models`    |
| `--output_dir` | Directory to save CSV results and plots | `output`               |
| `--lime_samples` | Number of perturbations for LIME (higher = more accurate but slower) | `500`                  |
| `--max_examples` | Max number of texts to analyze for aggregate reports | `797`                  |

**Example: High-Fidelity Run with Custom Paths**

```bash
python analysis.py \
  --data_dir /path/to/my/data \
  --model_dir /path/to/my/models \
  --lime_samples 1000 \
  --output_dir ./final_results
```
-----------------------------

## Data Format Details
You can read the data files using pandas as follows:

```python
import pandas as pd
df = pd.read_csv("implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv", delimiter="\t")
```
-----------------------------







