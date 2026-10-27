# EEG Motor Imagery Classification

Reproducible motor-imagery decoding experiments on BCI Competition IV Dataset
2a. The repository compares EEGNet temporal aggregation heads and evaluates
generalization within sessions, across sessions and across subjects.

## Main findings

- Replacing the flattened EEGNet classifier with temporal mean pooling improved
  mean within-session accuracy from 54.51% to 60.14%.
- Learned temporal attention added only a small additional gain, reaching 60.91%
  with position-aware residual attention.
- Source-only nested leave-one-subject-out accuracy was 41.63%. Transductive
  unsupervised Euclidean Alignment increased it to 50.88%.

## Layout

```text
eeg_bci/
├── configs/default.yaml
├── data/raw/                 # Put A01T.mat ... A09T.mat here
├── data/evaluation_labels/   # Official A01E.mat ... A09E.mat labels
├── scripts/
│   ├── inspect_data.py
│   ├── train_cv.py
│   ├── train_cross_session.py
│   └── train_cross_subject.py
├── src/eeg_bci/
│   ├── data.py               # MATLAB trial extraction
│   ├── preprocessing.py      # filtering, resampling, standardization
│   ├── alignment.py          # Euclidean Alignment
│   ├── datasets.py           # PyTorch dataset/loaders
│   ├── models/               # EEGNet and temporal attention
│   ├── training.py
│   └── evaluation.py
└── artifacts/                # Generated cache, models, metrics and plots
```

## Setup

```bash
cd eeg_bci
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and tests, install `requirements-dev.txt` and run `pytest`.

Copy training files into `data/raw/`, then inspect them:

```bash
python scripts/inspect_data.py
```

Run stratified four-fold cross-validation:

```bash
python scripts/train_cv.py --model eegnet --subjects A01
python scripts/train_cv.py --model attention --subjects A01
python scripts/train_cv.py --model temporal-mean --subjects A01
```

Use `--subjects all` for A01–A09. Model architecture and input window are
independent experiment variables:

```bash
python scripts/train_cv.py --model eegnet --window wide --subjects all
python scripts/train_cv.py --model attention --window wide --subjects all
```

`standard` is the 2.5–4.5 s window and `wide` is 2–6 s. The deprecated
`attention-wide` model name remains as a compatibility alias for
`--model attention --window wide`.

For a stronger within-session evaluation, keep complete recording runs out of
the training set:

```bash
python scripts/train_cv.py \
  --model attention \
  --window standard \
  --split-strategy run \
  --subjects all \
  --run-name attention_standard_run_cv
```

`trial` is the original stratified trial-wise split. `run` uses grouped folds,
so trials from the same recording run cannot appear in both training and test
sets. Neither mode is a cross-session or cross-subject evaluation; those require
separate session files or subject-level outer folds.

All standardization statistics are learned only from each training fold.
Preprocessed caches are keyed by the full preprocessing configuration and the
source file identity. Every experiment is written to its own directory under
`artifacts/runs/` with:

- the complete configuration and runtime manifest;
- fold, subject-level and overall metrics;
- exact train/validation/test indices and test run IDs;
- checkpoints, predictions, attention weights and figures.

Existing non-empty experiment directories are never overwritten. Supply a new
`--run-name` when repeating an experiment with changed settings.

Compare completed runs with:

```bash
python scripts/compare_runs.py \
  artifacts/runs/eegnet_standard_trial \
  artifacts/runs/mean_standard_trial \
  artifacts/runs/attention_standard_trial \
  --output artifacts/model_comparison.csv
```

The same commands work in Colab after cloning the repository. Place the MAT
files in the expected directory or pass `--data-dir /content/drive/...`.

## Follow-up temporal experiments

The original attention model sees 8 temporal bins in the standard window. The
high-resolution variants retain 16 bins (32 for the wide window), enabling a
clean mean-pooling versus attention comparison at the same resolution:

```bash
python scripts/train_cv.py --model temporal-mean-hires --window standard --subjects all
python scripts/train_cv.py --model attention-hires --window standard --subjects all
python scripts/train_cv.py --model attention-positional-residual --window standard --subjects all
```

`attention-positional-residual` adds learned temporal position information and
starts close to mean pooling through a learnable residual gate. This tests
whether attention helps without forcing the network to abandon the more stable
mean representation.

Use `--seed` to repeat a run. To execute a full matrix over models, windows,
split strategies and seeds:

```bash
python scripts/run_experiment_suite.py \
  --models temporal-mean-hires attention-hires attention-positional-residual \
  --windows standard wide --split-strategies trial run \
  --seeds 42 43 44 --subjects all
```

After an attention run, quantify how concentrated and trial-specific its
weights are:

```bash
python scripts/analyze_attention.py artifacts/runs/attention-hires_standard_trial_seed42
```

Cross-session evaluation requires the separate `A01E.mat`–`A09E.mat` files and
their official labels. It should not be reported from the training-session
files alone.

## Cross-session evaluation

Place `A01T.mat`–`A09T.mat` and `A01E.mat`–`A09E.mat` in `data/raw/`. Place the
official evaluation labels in `data/evaluation_labels/` using the corresponding
`A01E.mat`–`A09E.mat` filenames. Train and select the model only on the T
session, then evaluate once on the E session:

```bash
python scripts/train_cross_session.py \
  --model temporal-mean-hires \
  --window standard \
  --subjects all \
  --seed 42 \
  --run-name cross_session_mean_hires_seed42
```

Enable Euclidean Alignment with `--alignment ea`. Its source reference is
estimated from the T-session training split, while the target reference is
estimated from E-session EEG without labels:

```bash
python scripts/train_cross_session.py \
  --model temporal-mean-hires \
  --alignment ea \
  --window standard \
  --subjects all \
  --seed 42 \
  --run-name cross_session_mean_hires_ea_seed42
```

This is transductive unsupervised adaptation because the target EEG, but not its
labels, is used to estimate the target covariance reference.

## Cross-subject evaluation

Run nested leave-one-subject-out evaluation on the T sessions. The outer target
subject is excluded from training and model selection; inner `GroupKFold`
splits source participants for early-stopping epoch selection:

```bash
python scripts/train_cross_subject.py \
  --model temporal-mean-hires \
  --window standard \
  --inner-splits 4 \
  --seed 42 \
  --run-name temporal_mean_hires_standard_nested_loso_seed42
```

Run the matched Euclidean Alignment experiment with the same architecture:

```bash
python scripts/train_cross_subject.py \
  --model temporal-mean-hires \
  --alignment ea \
  --window standard \
  --inner-splits 4 \
  --seed 42 \
  --run-name temporal_mean_hires_standard_nested_loso_ea_seed42
```

Each source participant is aligned independently. The held-out target
participant is aligned using their unlabeled EEG, while target labels are used
only for final scoring. This is a transductive unsupervised adaptation setting,
not a source-only evaluation.
