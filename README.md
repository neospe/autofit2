
# autofit2

Few-shot text classification. Massively multilingual (50+ languages), fully automated pipeline built on `setfit` and SBERT embeddings.

## Key Features

*   **Few-Shot Learning:** High precision (95–99%) with a few dozen labeled examples.
*   **Multilingual Support:** Pretrained models for 20 languages; evaluation corpora for 50+. Scalable to 100+ via Common Crawl.
*   **Automated Pipeline:** End-to-end preprocessing, fine-tuning, evaluation, and deployment from a single JSON config.
*   **Reproducibility & Transparency:** JSON-based configuration, model card generation, and CO₂ emission tracking.

## Quickstart

Train an English moderation model on a 2% sample of [Davidson et al.](https://github.com/t-davidson/hate-speech-and-offensive-language), with one benchmark target:

```bash
git clone https://github.com/neospe/autofit2 && cd autofit2
pip install -r requirements.txt
git clone https://github.com/t-davidson/hate-speech-and-offensive-language data/en/hate-speech-and-offensive-language
cp examples/demo.json .  # the config is moved to history/ after the run
python train.py demo.json
```

Runs on CUDA, Apple silicon (MPS) or CPU; about 12 minutes on an M1. The base model (`all-MiniLM-L6-v2`) is downloaded on the first run.

Output:
*   `history/demo.log`: accuracy and F1 per target.
*   `models-out/demo-mod-en.json`: model card with benchmarks, self-consistency, emissions and a bias evaluation: on neutral sentences, one per identity or occupation term (`eval-data`), the false positive rate and the flagged terms, and each term's received attention relative to the base model (terms the fine-tuned model attends to much more are candidates for shortcuts).
*   `models-out/demo-mod-en.mar`: model archive for TorchServe.
*   `models-in/demo-mod-en`: the fine-tuned embedding, usable as a base model for custom targets.

Or in a container:

```bash
docker build -t autofit2 .
docker run --rm -v "$PWD":/service autofit2 python train.py demo.json
```

## Usage

**1. Prepare Data**
Use `dataload` or implement a custom loader providing labeled examples.

**2. Configure**
Create `myproject.json` specifying dataset paths, model settings, and output directories. Supports multi-language/task blocks.

**3. Run**

The pipeline supports resumable execution.

```bash
python train.py myproject.json
```

**4. Output**
*   Deployable model archive.
*   Generated model card (training details, intended use, performance metrics, bias evaluation).


## Configuration

`myproject.json` defines the training parameters. Its structure depends on the target type: **Base Models** (`all`) or **Custom Models** (`custom`).

### General Structure

```json
{
  "<task-key>": {
    "<language-key>": {
      "base": {
        "model file": "<path>",          // Relative path, no trailing slash (e.g. "models-in/all-MiniLM-L6-v2")
        "model type": "<string>",          // e.g., "bert"
        "pretraining task": "<string>",  // e.g., "sentence similarity"
        "downstream task": "<string>"    // e.g., "binary text classification"
      },
      "targets": {
        "<id-key>": { ... }             // See Target Options below
      }
    }
  }
}
```

### Target Types

The `"targets"` dictionary supports three specific key types:

1.  **`all`** (Base Model)
    *   Generates a full set of artifacts: model folder, archive, and card.
    *   **Model ID:** Derived from the config filename (`{config_name}-{task}-{lang}`). The config filename must be stable.
2.  **`custom`** (Custom Model)
    *   Generates a full set of artifacts: model folder, archive, and card.
    *   **Model ID:** can be auto-generated as a 14–16 character lowercase alphanumeric string.
3.  **`benchmark 1..N`** (Benchmarking Only)
    *   Does **not** generate model artifacts.
    *   Outputs only score logs.
    *   Must be used in conjunction with an `all` target to produce output.

### Target Options

Each entry in the `"targets"` dictionary supports `description`, `link`, `loader` and an optional `options` object:

```json
"custom": {
  "options": {"train embedding": false, "base clf": "98sdfkjhw3r8sf"},
  "loader": ["en_hate1()"]
}
```

| Key | Type | Description |
| :--- | :--- | :--- |
| `description` | `string` | Free-form description of the target. |
| `link` | `string` | URL to source data or documentation. |
| `options` → `train embedding` | `bool` | Fine-tune embeddings during training (default: `true`). |
| `options` → `base clf` | `string` | ID string pointing to a `.joblib` file located in `BASE_PATH`. Must match exactly. |
| `options` → `sample ratio` | `float` | Random sample of total data for full training (e.g., `0.5` = 50%). |
| `options` → `embedding sample ratio` | `float` | Random sample of data used **only** for embedding fine-tuning (e.g., `0.1` = 10%). |
| `options` → `mask shortcuts` | `bool` | Detect-then-mask (default: `false`): after a first fine-tune, the 50 words whose received attention grew most are checked against wordlists of legitimate offense cues in `wordlists/<lang>/` (plain text, one word per line, e.g. profanity, slur and insult lists); identity terms from `eval-data/<lang>-terms.txt` never count as legitimate. The other words are removed from the training texts and the embedding is fine-tuned again (about twice the training time). The model card lists masked and kept words. Removes shortcut words; it cannot fix bias in the labels. |
| `options` → `attention regularization` | `bool` | Entropy-based attention regularization (EAR) during embedding fine-tuning (default: `false`). It flattens attention but does not keep the model from using shortcut tokens, and it costs pretrained information. |

### Loaders

The `"loader"` field defines how data is ingested and transformed. It expects a list of commands (functions or transformations):

```json
"loader": ["command_1", "command_2"]
```

*   **Command Definition:** Each command must return a list of dictionaries with keys `text` and `label`. Commands can be raw loader functions or wrapped transformations (e.g., list comprehensions, lambdas).
*   **Data Splitting Logic:**
    *   **If 2 commands AND target != `all`:**
        *   Command 1 → Training Data
        *   Command 2 → Evaluation Data
    *   **Else (Target = `all`):**
        *   All commands are concatenated into a single dataset.
        *   Split: **100/100** (No split; entire set used for training).
    *   **Else (Other Targets, e.g., `custom` or benchmarks with 1 command):**
        *   All commands are concatenated into a single dataset.
        *   Split: **70/30** (Train/Test).

## Configuration Example

```json
{
  "mod": {
    "el": {
      "base": {
        "model file": "models-in/paraphrase-multilingual-MiniLM-L12-v2",
        "model type": "bert",
        "pretraining task": "sentence similarity",
        "downstream task": "binary text classification"
      },
      "targets": {
        "benchmark 1": {
          "description": "Pitenis et al. - Offensive Language Identification in Greek",
          "link": "https://arxiv.org/abs/2003.07459",
          "loader": [
            "el_offense20(files=['offenseval2020-greek/offenseval-gr-training-v1/offenseval-gr-training-v1.tsv'])",
            "el_offense20(files=['offenseval2020-greek/offenseval-gr-testsetv1/offenseval-gr-test-v1-combined.tsv'])"
          ]
        },
        "all": {
          "loader": [
            "el_offense20()"
          ]
        }
      }
    }
  }
}
```

### Breakdown: Finetuning a Sentence Transformer

To fine-tune a base model for a specific task and language, define a config block like the one below. This example sets up a text moderation (mod) pipeline for Greek (el) using a multilingual sentence transformer.

**Base Model Setup**

```json
"base": {
  "model file": "models-in/paraphrase-multilingual-MiniLM-L12-v2",
  "model type": "bert",
  "pretraining task": "sentence similarity",
  "downstream task": "binary text classification"
}
```
- Model file: Path to the pretrained transformer.
- Model type: Architecture type (e.g., BERT).
- Pretraining task: Original task the model was trained on.
- Downstream task: Task you're adapting it to (e.g., moderation, sentiment analysis).

**Targets**

You can specify multiple finetuning targets. Each target defines a dataset and training strategy.

1. `benchmark 1`
```json
"benchmark 1": {
  "description": "Pitenis et al. - Offensive Language Identification in Greek",
  "link": "https://arxiv.org/abs/2003.07459",
  "loader": [
    "el_offense20(files=['offenseval2020-greek/offenseval-gr-training-v1/offenseval-gr-training-v1.tsv'])",
    "el_offense20(files=['offenseval2020-greek/offenseval-gr-testsetv1/offenseval-gr-test-v1-combined.tsv'])"
  ]
}
```
- Uses a train/test split for evaluation.
- Based on a published benchmark dataset.

2. `all`
```json
"all": {
  "loader": ["el_offense20()"]
}
```
- Uses the full dataset for training.
- No explicit evaluation—this is for production-grade finetuning.

