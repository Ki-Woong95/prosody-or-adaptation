# Paper results

All WER values are percentages. Means and sample standard deviations are over three training seeds.

## Recognition performance

| Corpus | Baseline | Null | Learned | Learned − Null | 95% CI | Null − Baseline | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buckeye | 37.27 ± 0.10 | **35.82 ± 0.13** | 35.89 ± 0.11 | +0.069 | [−0.222, +0.341] | **−1.447** | [−1.929, −0.942] |
| Switchboard | 31.95 ± 0.09 | 31.24 ± 0.04 | **31.15 ± 0.07** | −0.090 | [−0.209, +0.027] | **−0.714** | [−0.855, −0.584] |
| AMI IHM | 39.11 ± 0.22 | **38.23 ± 0.22** | 38.23 ± 0.20 | +0.004 | [−0.269, +0.275] | **−0.885** | [−1.238, −0.541] |

Holm-corrected Learned−Null comparisons are not significant on any corpus (`p >= .400`). Null−Baseline is significant on all three corpora (`pHolm < .001`).

The primary conclusion is therefore narrow: **under this frozen, post-hoc fusion architecture, the learned auxiliary representation provides little measurable incremental WER benefit over a parameter-matched zero-input adapter, while the adapter pathway itself produces a robust improvement over Baseline.**

## Phase-1 encoder

The selected CASPER checkpoint achieves:

| Metric | Value |
|---|---:|
| F0 correlation | 0.8885 |
| F0 MAE | 146.6 cents |
| Voicing F1 | 0.8903 |
| ΔF0 correlation | 0.7031 |
| Energy correlation | 0.9982 |
| Spectral-tilt correlation | 0.9983 |

These metrics establish that the encoder learned its supervision targets; they do not imply that its 64-D hidden representation contains only prosodic information.

## Feature interventions

The trained Learned model is re-evaluated without retraining after modifying its auxiliary representation. Values below are WER increases relative to the unmodified Learned condition.

| Corpus | Time shuffle | Utterance shuffle | Zero |
|---|---:|---:|---:|
| Buckeye | +0.17 | +0.42 | +4.85 |
| Switchboard | +0.30 | +1.06 | +16.93 |
| AMI IHM | +0.35 | +1.01 | +12.77 |

The large zero-input degradation shows that the trained Learned model becomes dependent on its auxiliary pathway. It does **not** measure the causal benefit of the representation, because the separately trained Null system reaches essentially the same WER without that representation.

## Adapter behavior

Both Null and Learned produce substantial residual modifications of frozen HuBERT states, especially in upper layers. Mean relative residual magnitude is larger for Null than Learned on all three corpora:

| Corpus | Learned | Null |
|---|---:|---:|
| Buckeye | 0.249 | 0.351 |
| Switchboard | 0.527 | 0.825 |
| AMI IHM | 0.673 | 1.091 |

This is consistent with the recognition results: the zero-input control is not an inactive adapter. It learns a substantial transformation of the frozen representation.

## Machine-readable files

- `paper_results.json`: seed-level WER and parameter counts
- `paper_inference.json`: paired hierarchical bootstrap for primary and secondary comparisons
- `feature_interventions.json`: test-time feature interventions
- `intervention_inference.json`: paired inference for intervention contrasts
- `residual_analysis.json`: layerwise gate and residual statistics
