# Automated Essay Scoring 2.0 baseline

This repository contains a deterministic CPU baseline for the AI Coding Gym
version of Learning Agency Lab — Automated Essay Scoring 2.0.

The baseline uses five-fold stratified cross-validation, fold-local word and
character TF-IDF features, Ridge regression, and out-of-fold threshold tuning
for the quadratic weighted kappa (QWK) competition metric. Fitting each
vectorizer on its training fold prevents validation-text leakage.

## Run

The project uses the existing Conda environment named `1011`:

```bash
conda run -n 1011 python -m pip install -r requirements.txt
conda run -n 1011 python train.py
```

This produces:

- `submission.csv`: integer scores from 1 through 6 in test-file order.
- `baseline_metrics.json`: configuration, per-fold metrics, OOF QWK,
  optimized thresholds, and prediction distribution.

Submit the predictions with:

```bash
conda run -n 1011 aicodinggym mle submit \
  learning-agency-lab-automated-essay-scoring-2 \
  -F submission.csv \
  -m "5-fold word+char TF-IDF Ridge baseline with OOF thresholds"
```

## Stage 2: engineered feature blend

The second stage adds a separate nonlinear model over 56 interpretable essay
features. They cover length, sentence and paragraph structure, lexical
diversity, punctuation, discourse markers, pronouns, readability, and simple
writing-error proxies. Keeping this branch separate lets the model learn
nonlinear relationships (for example, essay length has diminishing returns)
without densifying the large TF-IDF matrix.

```bash
conda run -n 1011 python train_stage2.py
conda run -n 1011 python -m unittest discover -s tests -v
```

The script creates `stage2_submission.csv` and `stage2_metrics.json`. Both the
blend weight and ordinal score thresholds are selected only from aligned OOF
predictions. The test predictions are untouched until those parameters are
fixed.
