"""Regression tests for per-utterance prosody-to-frame alignment in Phase 2.

The alignment path is tested independently of HuBERT's convolutional frontend so
padding effects in the frozen backbone do not obscure the prosody invariant.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers")
import torch
from torch import nn

from prosody_adaptation.asr_model import ProductionASR, align_prosody_to_frames
from prosody_adaptation.lengths import frame_mask, hubert_output_lengths, prosody_frame_count
from prosody_adaptation.prosody_model import ProsodyEncoderWithHeads

SAMPLE_RATE = 16_000
HIDDEN_DIM = 24
SHORT, LONG = 2 * SAMPLE_RATE, 5 * SAMPLE_RATE


def _lengths(*samples):
    return torch.tensor(list(samples), dtype=torch.long)


# ------------------------------------------------- 1. aligned prosody identity


def _encoder(seed=0):
    torch.manual_seed(seed)
    return ProsodyEncoderWithHeads(
        channels=16, gru_hidden=8, representation_dim=8
    ).encoder.requires_grad_(False).eval()


def test_aligned_prosody_is_identical_alone_and_beside_a_longer_neighbour():
    encoder = _encoder()
    torch.manual_seed(1)
    short = torch.randn(1, SHORT)
    long = torch.randn(1, LONG)
    short_frames = int(hubert_output_lengths(_lengths(SHORT))[0])
    long_frames = int(hubert_output_lengths(_lengths(LONG))[0])

    with torch.no_grad():
        alone, alone_lengths = encoder(short, _lengths(SHORT))
        padded = torch.cat((torch.cat((short, torch.zeros(1, LONG - SHORT)), 1), long), 0)
        batched, batched_lengths = encoder(padded, _lengths(SHORT, LONG))

    assert int(alone_lengths[0]) == int(batched_lengths[0]) == prosody_frame_count(SHORT)
    aligned_alone = align_prosody_to_frames(
        alone, alone_lengths, _lengths(short_frames), short_frames
    )
    aligned_batched = align_prosody_to_frames(
        batched, batched_lengths, _lengths(short_frames, long_frames), long_frames
    )
    difference = (aligned_alone[0] - aligned_batched[0, :short_frames]).abs().max()
    torch.testing.assert_close(
        aligned_alone[0], aligned_batched[0, :short_frames], atol=1e-5, rtol=1e-5
    )
    # The residue is packed-GRU float non-associativity across batch shapes, not
    # alignment: the whole-batch interpolation this replaced differed by ~1.2 on
    # features of scale ~2, six orders of magnitude larger.
    assert difference < 1e-5


def test_aligned_prosody_padding_is_exactly_zero():
    encoder = _encoder()
    torch.manual_seed(2)
    waveforms = torch.cat(
        (torch.cat((torch.randn(1, SHORT), torch.zeros(1, LONG - SHORT)), 1),
         torch.randn(1, LONG)), 0,
    )
    lengths = _lengths(SHORT, LONG)
    frames = hubert_output_lengths(lengths)
    width = int(frames.max())
    with torch.no_grad():
        prosody, prosody_lengths = encoder(waveforms, lengths)
    aligned = align_prosody_to_frames(prosody, prosody_lengths, frames, width)

    assert aligned.shape == (2, width, 8)
    assert torch.equal(aligned[0, int(frames[0]):], torch.zeros(width - int(frames[0]), 8))
    assert aligned[0, : int(frames[0])].abs().sum() > 0


def test_alignment_uses_each_utterance_own_length_not_the_batch_width():
    """A ramp makes the mapping readable: output k must read source coordinate
    ``(k+0.5)*N/T-0.5`` for that utterance's own N and T, never the batch's."""
    source_frames, target_frames, width = 4, 7, 11
    ramp = torch.arange(source_frames, dtype=torch.float32).reshape(1, source_frames, 1)
    padded = torch.cat((ramp, torch.full((1, 6, 1), 99.0)), dim=1)

    aligned = align_prosody_to_frames(
        padded, _lengths(source_frames), _lengths(target_frames), width
    )
    index = np.arange(target_frames)
    expected = np.clip((index + 0.5) * source_frames / target_frames - 0.5, 0, source_frames - 1)
    np.testing.assert_allclose(
        aligned[0, :target_frames, 0].numpy(), expected, rtol=0, atol=1e-5
    )
    assert torch.equal(aligned[0, target_frames:], torch.zeros(width - target_frames, 1))


def test_alignment_rejects_degenerate_and_overflowing_lengths():
    prosody = torch.randn(1, 5, 3)
    with pytest.raises(ValueError, match="must be positive"):
        align_prosody_to_frames(prosody, _lengths(0), _lengths(4), 4)
    with pytest.raises(ValueError, match="must be positive"):
        align_prosody_to_frames(prosody, _lengths(5), _lengths(0), 4)
    with pytest.raises(ValueError, match="batch is"):
        align_prosody_to_frames(prosody, _lengths(5), _lengths(9), 4)
    with pytest.raises(ValueError, match="agree on batch size"):
        align_prosody_to_frames(prosody, _lengths(5, 5), _lengths(4), 4)


# ------------------------------------------------ 2-5. end-to-end ProductionASR


class _PaddingInvariantHubert(nn.Module):
    """Stand-in whose hidden states depend only on (layer, frame index).

    Frame t of layer l is the same vector no matter how the batch is padded, so
    any residual variation in the model output comes from the prosody path.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, layers=13, max_frames=1024):
        super().__init__()
        generator = torch.Generator().manual_seed(99)
        self.register_buffer(
            "table", torch.randn(layers, max_frames, hidden_dim, generator=generator)
        )
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, input_values, attention_mask=None, output_hidden_states=True,
                return_dict=True):
        batch, width = input_values.shape[0], int(
            hubert_output_lengths(torch.tensor([input_values.shape[1]]))[0]
        )
        states = [self.table[layer, :width].expand(batch, -1, -1) for layer in range(13)]
        return type("Output", (), {"hidden_states": tuple(states)})()

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return hubert_output_lengths(lengths)


def _production_model(monkeypatch, tmp_path, condition):
    """Build the real ProductionASR with a padding-invariant backbone.

    ``transformers`` is a ``_LazyModule``, so patching the module attribute does
    not bind for ``from transformers import HubertModel``; patch the classmethod.
    """
    from transformers import HubertModel

    monkeypatch.setattr(
        HubertModel, "from_pretrained",
        staticmethod(lambda *args, **kwargs: _PaddingInvariantHubert()),
    )
    checkpoint = tmp_path / "prosody.pt"
    torch.manual_seed(7)
    torch.save({"model": ProsodyEncoderWithHeads(representation_dim=8).state_dict()}, checkpoint)
    config = {
        "hubert_model": "stub", "hubert_revision": "stub", "hidden_dim": HIDDEN_DIM,
        "prosody_dim": 8, "project_dim": 10, "blstm_hidden": 6, "blstm_layers": 2,
        "dropout": 0.0,
    }
    torch.manual_seed(8)
    model = ProductionASR(
        config, vocabulary_size=5, condition=condition,
        prosody_checkpoint=str(checkpoint) if condition != "ab1" else None,
    ).eval()
    if condition != "ab1":  # move off the identity initialization so fusion is live
        with torch.no_grad():
            for adapter in model.fusion.adapters:
                adapter.residual_scale.fill_(0.8)
                nn.init.normal_(adapter.gate.weight, std=0.3)
                nn.init.normal_(adapter.film.weight, std=0.05)
    return model


def _run(model, waveform_lengths, waveforms):
    attention_mask = torch.zeros(waveforms.shape, dtype=torch.long)
    for index, length in enumerate(waveform_lengths.tolist()):
        attention_mask[index, :length] = 1
    with torch.no_grad():
        return model(waveforms, attention_mask, waveform_lengths)


def _alone_and_padded(model):
    torch.manual_seed(3)
    short = torch.randn(1, SHORT)
    alone = _run(model, _lengths(SHORT), short)
    padded_batch = torch.cat(
        (torch.cat((short, torch.zeros(1, LONG - SHORT)), 1), torch.randn(1, LONG)), 0
    )
    batched = _run(model, _lengths(SHORT, LONG), padded_batch)
    return alone, batched


@pytest.mark.parametrize("condition", ["full", "ab3", "ab1"])
def test_valid_frame_logits_are_padding_invariant(monkeypatch, tmp_path, condition):
    model = _production_model(monkeypatch, tmp_path, condition)
    alone, batched = _alone_and_padded(model)
    valid = int(alone["frame_lengths"][0])
    torch.testing.assert_close(
        alone["logits"][0, :valid], batched["logits"][0, :valid], atol=1e-5, rtol=1e-5
    )


def test_whole_batch_alignment_breaks_padding_invariance(monkeypatch, tmp_path):
    """Whole-batch interpolation is not a valid substitute for per-utterance alignment."""
    import torch.nn.functional as F

    model = _production_model(monkeypatch, tmp_path, "full")

    def whole_batch(prosody, prosody_lengths, frame_lengths, width):
        return F.interpolate(
            prosody.transpose(1, 2), size=width, mode="linear", align_corners=False
        ).transpose(1, 2)

    monkeypatch.setattr("prosody_adaptation.asr_model.align_prosody_to_frames", whole_batch)
    alone, batched = _alone_and_padded(model)
    valid = int(alone["frame_lengths"][0])
    difference = (alone["logits"][0, :valid] - batched["logits"][0, :valid]).abs().max()
    assert difference > 1e-4, "whole-batch interpolation should be batch dependent"


def test_ab3_logits_are_identical_to_the_prosody_free_reference(monkeypatch, tmp_path):
    """AB3 zeroes the prosody contribution, so alignment cannot reach its output."""
    model = _production_model(monkeypatch, tmp_path, "ab3")
    torch.manual_seed(3)
    waveform = torch.randn(1, SHORT)
    reference = _run(model, _lengths(SHORT), waveform)["logits"]

    # Same input, but a completely different frozen Phase-1 encoder.
    torch.manual_seed(4242)
    replacement = ProsodyEncoderWithHeads(representation_dim=8).encoder.eval()
    model.prosody_encoder = replacement.requires_grad_(False)
    swapped = _run(model, _lengths(SHORT), waveform)["logits"]
    torch.testing.assert_close(reference, swapped, atol=0, rtol=0)

    # And it survives padding unchanged.
    padded = torch.cat((waveform, torch.zeros(1, LONG - SHORT)), 1)
    valid = int(hubert_output_lengths(_lengths(SHORT))[0])
    batched = _run(model, _lengths(SHORT), padded)["logits"]
    torch.testing.assert_close(reference[0, :valid], batched[0, :valid], atol=1e-6, rtol=1e-6)


def test_ab1_never_touches_the_prosody_path(monkeypatch, tmp_path):
    model = _production_model(monkeypatch, tmp_path, "ab1")
    assert model.prosody_encoder is None and model.fusion is None

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("AB1 must not align prosody")

    monkeypatch.setattr("prosody_adaptation.asr_model.align_prosody_to_frames", explode)
    alone, batched = _alone_and_padded(model)
    valid = int(alone["frame_lengths"][0])
    torch.testing.assert_close(
        alone["logits"][0, :valid], batched["logits"][0, :valid], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("condition", ["full", "ab3", "ab1"])
def test_valid_frame_lengths_and_masks_are_unchanged(monkeypatch, tmp_path, condition):
    model = _production_model(monkeypatch, tmp_path, condition)
    alone, batched = _alone_and_padded(model)

    expected = hubert_output_lengths(_lengths(SHORT, LONG))
    assert int(alone["frame_lengths"][0]) == int(expected[0])
    torch.testing.assert_close(batched["frame_lengths"], expected)
    # The short utterance's own frame count must not depend on its neighbour.
    assert int(alone["frame_lengths"][0]) == int(batched["frame_lengths"][0])
    # And the mask derived from it is the same in both settings.
    width = int(expected[0])
    torch.testing.assert_close(
        frame_mask(alone["frame_lengths"], width),
        frame_mask(batched["frame_lengths"][:1], width),
    )
    assert alone["logits"].shape[1] == width


def test_prosody_and_frame_lengths_stay_consistent_across_lengths():
    """Every utterance must have positive prosody and output frames to align."""
    for seconds in (1, 2, 5, 15, 20):
        samples = seconds * SAMPLE_RATE
        prosody = prosody_frame_count(samples)
        frames = int(hubert_output_lengths(_lengths(samples))[0])
        assert prosody > 0 and frames > 0
        # The two grids stay within one frame of each other, as a 20 ms hop implies.
        assert abs(prosody - frames) <= 1


def test_frozen_hubert_backbone_is_not_padding_invariant():
    """Documented, out of scope: HuBERT-base normalizes conv features over time.

    ``feat_extract_norm="group"`` is GroupNorm with one group per channel, i.e.
    InstanceNorm across the time axis, so the padded region shifts the statistics.
    This predates and is independent of the prosody path; it is recorded here so
    the invariance tests above are not mistaken for a whole-model guarantee.
    """
    from transformers import HubertConfig, HubertModel

    config = HubertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, conv_dim=(8, 8, 8, 8, 8, 8, 8),
        conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="group", do_stable_layer_norm=False,
    )
    assert config.feat_extract_norm == "group"
    torch.manual_seed(5)
    model = HubertModel(config).eval()
    short = torch.randn(1, SHORT)
    padded = torch.cat((short, torch.zeros(1, LONG - SHORT)), 1)
    valid = int(model._get_feat_extract_output_lengths(torch.tensor([SHORT])))
    with torch.no_grad():
        alone = model(short, output_hidden_states=True).hidden_states[-1]
        batched = model(padded, output_hidden_states=True).hidden_states[-1]
    assert (alone[0, :valid] - batched[0, :valid]).abs().max() > 1e-3
