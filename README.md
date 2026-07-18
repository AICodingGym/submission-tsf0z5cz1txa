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

## Stage 3: ModernBERT cross-validation

Stage 3 uses the fixed duplicate-aware folds from `prepare_stage3.py`, an
offline-pinned ModernBERT-base snapshot, and a token cache. The selected pilot
configuration is a bounded regression head, 1,024 tokens, and two epochs. The
multitask ordinal head and 1,280-token variant were stopped after the pilot
because neither improved fold-0 QWK while both added complexity or memory use.

Run the reproducible preparation and pilot comparison with:

```bash
conda run -n 1011 python prepare_stage3.py
conda run -n 1011 python cache_tokens.py
conda run -n 1011 python evaluate_stage3_pilots.py \
  pilot_reg_1024 pilot_multi_1024 pilot_multi_1280 --fold 0
```

The production folds must run sequentially because simultaneous 3090 training
caused excessive chassis heat. The runner checks completed artifacts before
skipping a fold, waits for cooldown, forces offline model loading, and stops a
training process at 84 C:

```bash
conda run -n 1011 python run_stage3_folds.py \
  --run-name pilot_reg_1024 --folds 0 1 2 3 4 --gpu 7 \
  --mode regression --max-length 1024 --epochs 2
```

After all five folds are complete, build the honest cross-fitted three-stream
blend and submission:

```bash
conda run -n 1011 python merge_stage3.py \
  --run-name pilot_reg_1024 --submission stage3_submission.csv
```

`stage3_metrics.json` records the Transformer result, cross-fitted blend,
per-fold diagnostics, length/topic slices, optimism gap, and the full-CV gate.
Continue to a second Transformer only if the honest three-way blend improves
stage 2 by at least 0.005 QWK. Otherwise stop at the stronger stage-2 model.
