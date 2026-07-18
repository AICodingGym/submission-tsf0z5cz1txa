# Stage 3 implementation report

Completed on 2026-07-18 in Conda environment `1011`.

## Outcome

The selected stage-3 system is a five-fold ModernBERT regression model blended
with the stage-2 TF-IDF and engineered-feature streams. It passed the full-CV
gate and was submitted successfully.

| Metric | Result |
| --- | ---: |
| Stage-2 cross-fitted QWK | 0.803775 |
| ModernBERT OOF QWK, globally calibrated | 0.817261 |
| Stage-3 three-way cross-fitted QWK | 0.820717 |
| Honest improvement over stage 2 | +0.016941 |
| Full-OOF tuned QWK | 0.822974 |
| Full-OOF optimism gap | 0.002257 |
| AI Coding Gym submission score | **0.83271** |

The predeclared full-CV gate required at least +0.005 cross-fitted QWK. The
observed +0.016941 improvement passes it by a wide margin.

## 3.0: reproducible data and model foundation

- Generated a fixed five-fold manifest for all 15,576 training essays.
- Detected no exact duplicate pairs and two near-duplicate links. The four
  affected essays were grouped so that no duplicate group crosses folds.
- Added eight topic clusters to the fold audit and kept the score distribution
  nearly identical across folds.
- Saved aligned stage-2 train/test raw predictions for later blending.
- Used cross-fitted calibration: each held-out fold is scored with weights and
  thresholds learned only from the other four folds.
- Pinned `answerdotai/ModernBERT-base` at revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8` and trained offline.
- Model SHA256:
  `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.
- Tokenizer SHA256:
  `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1`.

The token cache showed that 1,024 tokens fully cover about 99.0% of essays.
Longer essays use head-tail truncation so that both the introduction and ending
remain visible.

## 3.1: pilot selection and stopping decisions

All pilots used fold 0, two epochs, the same seed/folds/model snapshot, and BF16.

| Pilot | Optimized QWK | RMSE | MAE | Local three-way QWK* |
| --- | ---: | ---: | ---: | ---: |
| Regression, 1,024 tokens | **0.822764** | **0.551709** | **0.426385** | **0.830158** |
| Multitask, 1,024 tokens | 0.822290 | 0.561560 | 0.433965 | 0.826387 |
| Multitask, 1,280 tokens | 0.819540 | 0.563934 | 0.434883 | 0.828904 |

\* Pilot thresholds and blend parameters are fitted on fold 0, so this column
is optimistic and was used only for relative selection.

The production configuration is pure regression, 1,024 tokens, two epochs,
batch size 16, encoder learning rate `1.5e-5`, and head learning rate `5e-5`.
The ordinal multitask branch was stopped because it did not improve QWK and had
worse RMSE/MAE. The 1,280-token branch was stopped because it was slower, used
more memory, and scored lower.

## 3.2: full five-fold training

| Fold | Rows | Rounded QWK | Optimized QWK | RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3,116 | 0.808379 | 0.822764 | 0.551709 |
| 1 | 3,115 | 0.797884 | 0.814891 | 0.565161 |
| 2 | 3,115 | 0.801567 | 0.820175 | 0.562580 |
| 3 | 3,115 | 0.809784 | 0.824554 | 0.553945 |
| 4 | 3,115 | 0.798196 | 0.815108 | 0.564664 |

The optimized fold scores span only 0.009663, with no failed fold. Validation
loss improved in epoch 2 for every fold.

Running multiple adjacent RTX 3090 cards at once initially raised temperatures
to 85--87 C. Production training therefore ran one fold at a time. The runner
waits for cooldown, checks the GPU every 10 seconds, stops at 84 C, validates
completed prediction arrays, and skips valid completed folds on restart. The
observed production peak was 80 C.

## 3.3: honest three-way calibration

The final full-OOF parameters are:

- TF-IDF text weight: 0.30
- Engineered-feature weight: 0.10
- ModernBERT weight: 0.60
- Thresholds: 1.804329, 2.613882, 3.412940, 4.159111, 4.975187

The cross-fitted Transformer weight ranged from 0.55 to 0.75 across held-out
folds. Every fold retained nonzero weight on the older branches, confirming that
their error patterns still provide useful diversity.

## Diagnostics and remaining risks

- Transformer residual/length correlation is 0.0052, so there is no material
  monotonic length bias after using 1,024 tokens and head-tail truncation.
- Predictions still shrink at rare extremes. Transformer MAE is 0.756 for score
  1 and 0.908 for score 6, versus about 0.39--0.53 for scores 2--5. Cross-fitted
  thresholds mitigate this, but rare-class calibration remains the main risk.
- Topic-cluster cross-fitted QWK ranges from 0.698 to 0.848. Cluster 1 and
  cluster 6 are the weakest slices. Slice QWK is sensitive to the label range
  within each group, so these numbers identify audit targets rather than proving
  a causal topic problem.
- Length-quartile QWK rises from 0.562 to 0.699. This also reflects different
  within-slice label distributions and should not be interpreted as length alone
  causing the error gap.
- The full-OOF versus cross-fitted gap is only 0.002257, below the earlier 0.008
  risk threshold and slightly smaller than stage 2's 0.002830 gap.

## Stopping decision

The stage-3 system passes both the local pilot gate and the authoritative
five-fold gate, so it replaces stage 2 as the primary submission. A second
Transformer backbone is not started now: ModernBERT already contributed a
strong +0.016941 honest gain, the public score confirmed the direction, and a
second full five-fold run would add substantial heat and compute without a
specific diagnosed failure to address.

The next experiment should be opened only if another improvement cycle is
requested. It should target the rare 1/6 scores and weak topic slices, and must
demonstrate incremental cross-fitted value over the saved three-stream OOF
predictions before a full additional backbone is trained.

## Reproduction

```bash
conda run -n 1011 python prepare_stage3.py
conda run -n 1011 python cache_tokens.py
conda run -n 1011 python run_stage3_folds.py \
  --run-name pilot_reg_1024 --folds 0 1 2 3 4 --gpu 7 \
  --mode regression --max-length 1024 --epochs 2
conda run -n 1011 python merge_stage3.py \
  --run-name pilot_reg_1024 --submission stage3_submission.csv
conda run -n 1011 python -m unittest discover -s tests -v
```

The submitted CSV contains 1,731 unique test IDs in sample-submission order,
integer scores from 1 through 6, and SHA256
`f53ca6dc3927275f182e23b444801d1e3bd33450bcf23ee090efeeeae0894564`.
