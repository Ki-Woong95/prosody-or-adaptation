# Reproducibility

The released YAML files define the final experimental pipeline. The paper matrix contains 27 Phase-2 runs: three corpora × three conditions × three training seeds.

## Frozen components

- HuBERT: `facebook/hubert-base-ls960` at revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`
- Processor/tokenizer: `facebook/wav2vec2-base-960h` at revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- Phase-1 encoder: frozen throughout Phase 2
- Corpus partitions and transcript normalization: fixed across conditions

The Phase-1 encoder runs in FP32. Phase 2 may use mixed precision for trainable components.

Each run writes `runtime_metadata.json` with package versions, CUDA/PyTorch information, and hashes of relevant local inputs. The Git commit is stored separately in `git_commit.txt`. Training metrics are recorded locally in `training_log.jsonl`; no external experiment tracker is required.

## Experiment matrix

Paper-facing config names use:

```text
{corpus}_{baseline|null|learned}_seed{1|2|3}.yaml
```

where the internal condition IDs `ab1`, `ab3`, and `full` are retained for compatibility with the analysis code.

The run registry is `configs/results/paper_runs.yaml`.

## Result generation

After all registered runs exist under `outputs/`:

```bash
prosody-adaptation summarize-results
prosody-adaptation infer
prosody-adaptation analyze-features
prosody-adaptation infer-interventions
prosody-adaptation analyze-residuals
```

`summarize-results` computes seed-level WER summaries. `infer` uses paired test predictions and a hierarchical Poisson bootstrap; speaker resampling is used for Buckeye and AMI IHM, while Switchboard uses seed-utterance resampling because usable speaker labels are not present in the stored predictions. Holm correction is applied separately to the three Learned−Null tests and the three Null−Baseline tests.

Feature interventions evaluate the trained Learned checkpoints with the original representation, zeros, within-utterance time shuffling, and a representation from another utterance. These are post-hoc diagnostics and should not be interpreted as replacements for the separately trained Null condition.

`analyze-residuals` reports padding-masked layerwise gate and residual statistics for the Learned and Null checkpoints.

## Licensed inputs

Corpus audio, transcript-bearing manifests, checkpoints, and per-utterance predictions are not included. See `LICENSED_DATA.md` and `docs/DATA.md` for the frozen hashes and expected local paths.

## Resume behavior

Run directories are immutable. To resume an interrupted experiment, copy its YAML, choose a new `experiment` name, set `resume_checkpoint`, and launch the copied config. Dataset and normalization hashes must remain unchanged.
