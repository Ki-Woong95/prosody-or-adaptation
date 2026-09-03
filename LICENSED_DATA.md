# Licensed data

This repository does not redistribute corpus audio, verbatim transcripts, Phase-2 checkpoints trained on licensed speech, or per-utterance prediction files that contain transcript text. The Phase-1 CASPER prosody encoder is released separately on Hugging Face under CC BY-NC-SA 4.0.
## Public metadata

The repository includes only information needed to identify the frozen experimental inputs without reconstructing the corpora:

- Buckeye split metadata and manifest SHA-256
- Switchboard and AMI IHM split counts and manifest SHA-256
- CASPER dataset hashes and target-normalization statistics
- aggregate paper results and post-hoc statistics

## Buckeye

Rebuild the local manifest from a licensed corpus copy:

```bash
prosody-adaptation prepare buckeye \
  --raw-archive /path/to/buckeye/archive \
  --output data/buckeye_v2_1 \
  --config configs/data/buckeye_v2_1.yaml
```

The resulting `data/buckeye_v2_1/manifest.jsonl` must match the hash in `data/buckeye_v2_1/FROZEN.json`.

## Switchboard and AMI IHM

The fixed Arrow datasets and transcript-bearing manifests used in the experiments are not distributed. Their expected manifest paths and SHA-256 hashes are recorded in `configs/experiment/templates/` and the public descriptors under `data/`.

## Generated outputs

`outputs/` is intentionally ignored. It may contain checkpoints, resolved configurations, per-utterance predictions, and logs. Likewise, `results/interventions/` is ignored because intervention records may contain verbatim corpus transcripts. Cross-corpus Phase-1 `*.progress.jsonl` files are also ignored because they contain corpus segment identifiers; only aggregate metrics are released.
