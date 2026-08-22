# Architecture

The released system follows the final paper: a prosody-trained encoder is learned in Phase 1, frozen, and then used only as an auxiliary input to post-hoc adapters in Phase 2.

## Phase 1

```mermaid
flowchart LR
  W[CASPER waveform] --> M[80-bin log-mel\n50 ms window / 20 ms hop]
  M --> C[128-D projection +\n4 depthwise-separable conv blocks]
  C --> G[Bidirectional GRU]
  G --> R[64-D representation]
  R --> H[5 prediction heads]
  H --> L[Masked multi-task loss]
```

The heads predict log F0, voicing, Δlog F0, log energy, and spectral tilt. They are discarded after Phase 1; Phase 2 receives the 64-D hidden representation.

## Phase 2

```mermaid
flowchart LR
  A[Waveform] --> H[Frozen HuBERT-base]
  A --> P[Frozen Phase-1 encoder]
  H --> S[Saved states h0 ... h12]
  P --> R[64-D auxiliary representation]
  R --> Z{Condition}
  Z -->|Learned| F[12 post-hoc gated FiLM adapters]
  Z -->|Null: p = 0| F
  S --> F
  S -->|Baseline| W0[Learned weighted sum]
  F --> W[Learned weighted sum]
  W0 --> B[768 -> 512 projection\n2-layer BiLSTM + CTC]
  W --> B
```

HuBERT produces all hidden states before any adapter is applied. Each transformer-layer state is modified independently, so adapted layer `l` is not fed into HuBERT layer `l+1`. This is **post-hoc layerwise fusion**, not in-backbone injection.

### Conditions

- **Baseline (AB1):** no adapters.
- **Null (AB3):** all twelve adapters are present, but the auxiliary input is fixed to zero.
- **Learned (Full):** the same adapters receive the frozen 64-D representation.

Only Null and Learned are parameter matched. The Learned−Null contrast isolates the incremental value of the auxiliary representation; Null−Baseline measures the effect of adding the trainable adaptation pathway.

### Adapter

For a frozen HuBERT state `h` and aligned auxiliary representation `p`, each adapter computes FiLM parameters from normalized `p`, forms a candidate state, predicts a residual gate, and adds the gated residual to `h`. The learned residual scale is initialized at zero, so the module starts as an exact identity mapping.
