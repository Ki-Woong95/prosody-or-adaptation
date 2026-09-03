# Data

No corpus audio or verbatim transcripts are redistributed in this repository.

## Phase 1: CASPER

CASPER is used only to train and select the prosody-trained encoder. Audio is resampled to 16 kHz and divided into non-overlapping segments of at most 15 seconds. The paper uses a fixed 80/20 **segment-level** training/validation split with seed 42.

The released Phase-1 configuration expects:

```text
$PROSODY_ADAPTATION_CASPER_ROOT/
├── train_prosody_v2/
└── val_prosody_v2/
```

The final teacher and student frontends use the same non-centered 50 ms / 20 ms frame grid. CREPE outputs are aligned to that grid by timestamp. Frozen dataset and normalization hashes are stored in `configs/experiment/prosody_casper.yaml` and `manifests/casper/`.

## Phase 2: Buckeye

Buckeye uses 30/5/5 speaker-disjoint train/validation/test speakers, corresponding to 4,915/797/925 segments in the paper.

The public repository contains only:

```text
data/buckeye_v2_1/FROZEN.json
data/buckeye_v2_1/metadata.json
```

Regenerate the licensed manifest locally with `prosody-adaptation prepare buckeye`. Paper configs then read audio through the deterministic tar cache pointed to by `PROSODY_ADAPTATION_BUCKEYE_CACHE`.

## Phase 2: Switchboard and AMI IHM

The paper uses fixed local Arrow splits for Switchboard (Original data from:[Ho et al. 2005](https://www.isca-archive.org/interspeech_2025/ho25b_interspeech.html) and AMI IHM. Because those datasets and transcript-bearing manifests cannot be redistributed, the public repository includes only frozen descriptors with the expected manifest hashes and split sizes.

Expected local manifest paths are:

```text
data/switchboard/manifest.json
data/ami_ihm/manifest.json
```

The corresponding split sizes are:

| Corpus | Train | Validation | Test |
|---|---:|---:|---:|
| Switchboard | 185,402 | 20,601 | 51,501 |
| AMI IHM | 75,174 | 9,428 | 8,514 |

All conditions and seeds use the same frozen partitions. The training code checks the manifest SHA-256 before starting a paper run.
