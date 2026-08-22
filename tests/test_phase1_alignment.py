"""Regression tests for the canonical Phase-1 teacher/student frame grid."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from prosody_adaptation.lengths import (
    PROSODY_HOP_LENGTH,
    PROSODY_WINDOW_LENGTH,
    hubert_output_lengths,
    prosody_frame_centre_samples,
    prosody_frame_count,
)
from prosody_adaptation.phase1_data import TARGET_COLUMNS, collate_phase1
from prosody_adaptation.prosody_model import LogMelFrontend, ProsodyEncoderWithHeads
from prosody_adaptation.teacher import (
    CREPE_FRAME_OFFSET,
    HOP_LENGTH,
    SAMPLE_RATE,
    WINDOW_LENGTH,
    TeacherExtractor,
    _pitch_targets,
    _to_canonical_grid,
)

# Exact 15 s, several non-15-s tail chunks, and short legal chunks.
REPRESENTATIVE_LENGTHS = (
    15 * SAMPLE_RATE,       # exact 15 s chunk
    14 * SAMPLE_RATE + 7,   # tail, not a multiple of the hop
    9 * SAMPLE_RATE + 313,  # tail, arbitrary offset
    3 * SAMPLE_RATE,
    SAMPLE_RATE,            # shortest CASPER tail (MIN_TAIL_SAMPLES)
    WINDOW_LENGTH + 1,      # shortest length with two frames' worth of signal
    WINDOW_LENGTH,          # exactly one frame
)


# ---------------------------------------------------------------- A. frame grid


@pytest.mark.parametrize("length", REPRESENTATIVE_LENGTHS)
def test_canonical_frame_count_matches_student_frontend(length):
    frontend = LogMelFrontend(SAMPLE_RATE, 80, win_ms=50, hop_ms=20).eval()
    with torch.no_grad():
        mel, frame_lengths = frontend(
            torch.zeros(1, length), torch.tensor([length], dtype=torch.long)
        )
    assert mel.shape[1] == prosody_frame_count(length)
    assert int(frame_lengths[0]) == prosody_frame_count(length)


@pytest.mark.parametrize("length", REPRESENTATIVE_LENGTHS)
def test_teacher_mel_framing_is_bitwise_identical_to_student(length):
    """The strongest possible statement: same transform, same frames, same bits."""
    extractor = TeacherExtractor(crepe_model="tiny", device="cpu")
    frontend = LogMelFrontend(SAMPLE_RATE, 80, win_ms=50, hop_ms=20).eval()
    waveform = torch.from_numpy(
        np.random.default_rng(0).standard_normal(length).astype(np.float32)
    )[None]
    with torch.no_grad():
        teacher = extractor.mel(waveform)
        student = frontend.transform(waveform)
    assert teacher.shape == student.shape
    assert torch.equal(teacher, student)
    assert teacher.shape[2] == prosody_frame_count(length)


def test_frame_count_below_one_window_is_rejected():
    with pytest.raises(ValueError, match="shorter than"):
        prosody_frame_count(WINDOW_LENGTH - 1)


def test_canonical_frame_centres_are_hop_spaced_and_window_offset():
    centres = prosody_frame_centre_samples(4)
    np.testing.assert_array_equal(centres, [400.0, 720.0, 1040.0, 1360.0])
    assert PROSODY_HOP_LENGTH == HOP_LENGTH
    assert PROSODY_WINDOW_LENGTH == WINDOW_LENGTH


# ------------------------------------------------------- B. CREPE timestamping


def test_crepe_frame_offset_is_a_quarter_window():
    # CREPE(pad=True) frame i is centred at i*hop; canonical frame j at j*hop + win/2.
    assert CREPE_FRAME_OFFSET == pytest.approx(WINDOW_LENGTH / (2 * HOP_LENGTH))
    assert CREPE_FRAME_OFFSET == pytest.approx(1.25)


def test_crepe_resampling_aligns_by_timestamp_not_index():
    """A ramp makes the shift readable: canonical frame j must sample position j+1.25."""
    native = np.arange(20, dtype=np.float32)
    aligned = _to_canonical_grid(native, 10)
    np.testing.assert_allclose(aligned, np.arange(10) + 1.25, rtol=0, atol=1e-5)


@pytest.mark.parametrize("length", REPRESENTATIVE_LENGTHS)
def test_crepe_resampling_never_extrapolates(length):
    crepe_frames = 1 + length // HOP_LENGTH
    highest_query = (prosody_frame_count(length) - 1) + CREPE_FRAME_OFFSET
    assert highest_query <= crepe_frames - 1


def _stub_torchcrepe(monkeypatch, frequency_fn=None, periodicity_value=1.0):
    """Install a fake torchcrepe whose output is a known function of frame index."""

    def predict(audio, sample_rate, hop_length, fmin, fmax, model, decoder,
                return_periodicity, device, batch_size):
        frames = 1 + audio.shape[-1] // hop_length
        index = torch.arange(frames, dtype=torch.float32)
        frequency = frequency_fn(index) if frequency_fn else torch.full_like(index, 120.0)
        periodicity = torch.full_like(index, periodicity_value)
        return frequency[None], periodicity[None]

    module = types.ModuleType("torchcrepe")
    module.predict = predict
    module.decode = types.SimpleNamespace(viterbi=object())
    monkeypatch.setitem(sys.modules, "torchcrepe", module)


@pytest.mark.parametrize("length", REPRESENTATIVE_LENGTHS)
def test_every_teacher_target_lands_on_the_canonical_grid(monkeypatch, length):
    _stub_torchcrepe(monkeypatch)
    extractor = TeacherExtractor(crepe_model="tiny", device="cpu")
    waveform = np.random.default_rng(1).standard_normal(length).astype(np.float32)
    targets = extractor.extract(waveform)
    expected = prosody_frame_count(length)
    for name, values in targets.as_columns().items():
        assert len(values) == expected, name


def test_teacher_f0_track_carries_the_crepe_shift(monkeypatch):
    """F0 on canonical frame j must come from CREPE position j + 1.25, not j."""
    _stub_torchcrepe(monkeypatch, frequency_fn=lambda index: 100.0 + index)
    extractor = TeacherExtractor(crepe_model="tiny", device="cpu")
    length = 3 * SAMPLE_RATE
    waveform = np.random.default_rng(2).standard_normal(length).astype(np.float32)
    targets = extractor.extract(waveform)
    voiced = targets.voiced > 0.5
    index = np.arange(len(targets.log_f0))
    expected = np.log(100.0 + index + CREPE_FRAME_OFFSET)
    np.testing.assert_allclose(
        targets.log_f0[voiced], expected[voiced], rtol=0, atol=2e-6
    )


def test_teacher_rejects_a_mel_that_leaves_the_canonical_grid(monkeypatch):
    _stub_torchcrepe(monkeypatch)
    extractor = TeacherExtractor(crepe_model="tiny", device="cpu")
    monkeypatch.setattr(extractor, "mel", lambda audio: torch.zeros(1, 80, 3))
    with pytest.raises(ValueError, match="canonical grid"):
        extractor.extract(np.zeros(SAMPLE_RATE, dtype=np.float32))


# ------------------------------------------------------------ C. target masking


def test_pitch_targets_mask_unvoiced_and_non_transitions():
    frequency = np.array([100.0, 110.0, 121.0, 130.0, 20.0], dtype=np.float32)
    periodicity = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    log_energy = np.array([-9.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    log_f0, voiced, delta = _pitch_targets(frequency, periodicity, log_energy)

    # Frame 0 fails the energy gate, frame 4 falls below F0_MIN_HZ.
    np.testing.assert_array_equal(voiced, [0.0, 1.0, 1.0, 1.0, 0.0])
    # Delta is non-zero only where the previous frame was also voiced.
    assert delta[0] == 0.0 and delta[1] == 0.0 and delta[4] == 0.0
    np.testing.assert_allclose(delta[2], np.log(121.0) - np.log(110.0), atol=1e-6)
    np.testing.assert_allclose(delta[3], np.log(130.0) - np.log(121.0), atol=1e-6)
    assert log_f0[0] == 0.0 and log_f0[4] == 0.0


def _item(length, frames=None, voiced=None):
    frames = prosody_frame_count(length) if frames is None else frames
    item = {"waveform": np.zeros(length, dtype=np.float32)}
    for name in TARGET_COLUMNS:
        item[name] = np.arange(frames, dtype=np.float32)
    item["voiced_mask"] = (
        np.ones(frames, dtype=np.float32) if voiced is None else voiced.astype(np.float32)
    )
    return item


def test_collate_masks_padding_out_of_every_target():
    long, short = 3 * SAMPLE_RATE, SAMPLE_RATE
    batch = collate_phase1([_item(long), _item(short)])
    long_frames, short_frames = prosody_frame_count(long), prosody_frame_count(short)

    assert batch["valid_mask"].shape[1] == long_frames
    assert int(batch["valid_mask"][0].sum()) == long_frames
    assert int(batch["valid_mask"][1].sum()) == short_frames
    assert not batch["valid_mask"][1, short_frames:].any()
    # Padded frames must be exactly zero, and must never be flagged as transitions.
    assert torch.equal(batch["energy"][1, short_frames:], torch.zeros(long_frames - short_frames))
    assert not batch["voiced_transition_mask"][1, short_frames:].any()


def test_collate_transition_mask_requires_two_consecutive_voiced_frames():
    frames = prosody_frame_count(SAMPLE_RATE)
    voiced = np.zeros(frames, dtype=np.float32)
    voiced[[3, 4, 5, 9]] = 1.0
    batch = collate_phase1([_item(SAMPLE_RATE, voiced=voiced)])
    transitions = batch["voiced_transition_mask"][0]
    assert transitions.nonzero().flatten().tolist() == [4, 5]


def test_collate_rejects_targets_from_a_different_framing_convention():
    """A v1 (center=True) cache has 3 extra frames; that must fail loudly."""
    length = 15 * SAMPLE_RATE
    stale = _item(length, frames=1 + length // HOP_LENGTH)
    with pytest.raises(ValueError, match="canonical Phase-1 grid"):
        collate_phase1([stale])


# ------------------------------------------------------- D. padding invariance


def test_prosody_encoder_is_padding_invariant_across_representative_lengths():
    torch.manual_seed(3)
    model = ProsodyEncoderWithHeads(channels=16, gru_hidden=8, representation_dim=8).eval()
    waveform = torch.randn(1, 3 * SAMPLE_RATE)
    padded = torch.cat((waveform, torch.zeros(1, 5 * SAMPLE_RATE)), dim=1)
    lengths = torch.tensor([3 * SAMPLE_RATE])
    with torch.no_grad():
        alone = model(waveform, lengths)["representation"]
        batched = model(padded, lengths)["representation"]
    valid = prosody_frame_count(3 * SAMPLE_RATE)
    torch.testing.assert_close(alone[:, :valid], batched[:, :valid], atol=1e-6, rtol=1e-6)


def _phase2_chain_modules(seed=4):
    from torch import nn

    from prosody_adaptation.asr_model import PackedBLSTMHead
    from prosody_adaptation.fusion import LayerwisePostHocFusion

    torch.manual_seed(seed)
    encoder = ProsodyEncoderWithHeads(
        channels=16, gru_hidden=8, representation_dim=8
    ).encoder.requires_grad_(False).eval()
    fusion = LayerwisePostHocFusion(hidden_dim=12, prosody_dim=8, layers=2).eval()
    with torch.no_grad():  # move off the identity initialization so fusion is live
        for adapter in fusion.adapters:
            adapter.residual_scale.fill_(0.7)
            nn.init.normal_(adapter.gate.weight, std=0.2)
    return encoder, fusion, nn.Linear(12, 10).eval(), PackedBLSTMHead(10, 6, 2, 5, dropout=0.0).eval()


def test_phase2_chain_is_padding_invariant_over_valid_prosody_frames():
    """The valid Phase-2 path is invariant to padding from neighboring utterances."""
    import torch.nn.functional as F

    from prosody_adaptation.asr_model import frozen_fp32_forward

    encoder, fusion, projector, head = _phase2_chain_modules()
    short_samples, long_samples = 2 * SAMPLE_RATE, 5 * SAMPLE_RATE
    waveform = torch.randn(1, short_samples)
    lengths = torch.tensor([short_samples])
    hubert_frames = int(hubert_output_lengths(lengths)[0])
    frame_lengths = torch.tensor([hubert_frames])
    hidden_states = [torch.randn(1, hubert_frames, 12) for _ in range(3)]

    def chain(values):
        with torch.no_grad():
            prosody, prosody_lengths = frozen_fp32_forward(encoder, values, lengths)
            prosody = prosody[:, : int(prosody_lengths[0])]
            prosody = F.interpolate(
                prosody.transpose(1, 2), size=hubert_frames, mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            return head(projector(fusion(hidden_states, prosody)), frame_lengths)

    alone = chain(waveform)
    batched = chain(torch.cat((waveform, torch.zeros(1, long_samples - short_samples)), dim=1))
    torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-5)


def test_whole_batch_interpolation_would_be_padding_dependent():
    """Guard against replacing per-utterance alignment with padded-batch interpolation."""
    import torch.nn.functional as F

    from prosody_adaptation.asr_model import frozen_fp32_forward

    short_samples, long_samples = 5 * SAMPLE_RATE, 15 * SAMPLE_RATE
    short_prosody = prosody_frame_count(short_samples)
    long_prosody = prosody_frame_count(long_samples)
    short_hubert = int(hubert_output_lengths(torch.tensor([short_samples]))[0])
    long_hubert = int(hubert_output_lengths(torch.tensor([long_samples]))[0])

    # Analytic: where does output frame k read from, alone vs padded?
    index = np.arange(short_hubert, dtype=np.float64)
    divergence = float(
        np.abs(
            ((index + 0.5) * short_prosody / short_hubert - 0.5)
            - ((index + 0.5) * long_prosody / long_hubert - 0.5)
        ).max()
    )
    assert 0.5 < divergence < 1.0, "known sub-frame stretch changed magnitude"

    # Empirical: the same divergence at the fused-feature level, uncropped.
    encoder, fusion, _, _ = _phase2_chain_modules(seed=8)
    waveform = torch.randn(1, short_samples)
    lengths = torch.tensor([short_samples])
    hidden_states = [torch.randn(1, short_hubert, 12) for _ in range(3)]

    def uncropped(values, hubert_frames):
        with torch.no_grad():
            prosody, _ = frozen_fp32_forward(encoder, values, lengths)
            return F.interpolate(
                prosody.transpose(1, 2), size=hubert_frames, mode="linear",
                align_corners=False,
            ).transpose(1, 2)[:, :short_hubert]

    alone = uncropped(waveform, short_hubert)
    padded = uncropped(
        torch.cat((waveform, torch.zeros(1, long_samples - short_samples)), dim=1), long_hubert
    )
    assert (alone - padded).abs().max() > 1e-3, "expected a measurable feature difference"
    # AB3 is immune: the adapter input is zeroed regardless of what arrives.
    torch.testing.assert_close(
        fusion(hidden_states, alone, null_features=True),
        fusion(hidden_states, padded, null_features=True),
        atol=0, rtol=0,
    )


# ------------------------------------------------------------ E. AB3 invariance


def test_ab3_fusion_input_cannot_depend_on_the_phase1_representation():
    """Null zeroes the auxiliary input before it reaches the adapters."""
    from prosody_adaptation.fusion import LayerwisePostHocFusion

    torch.manual_seed(5)
    fusion = LayerwisePostHocFusion(hidden_dim=8, prosody_dim=4, layers=2).eval()
    with torch.no_grad():
        for adapter in fusion.adapters:
            adapter.residual_scale.fill_(0.9)
            torch.nn.init.normal_(adapter.gate.weight, std=0.5)
            torch.nn.init.normal_(adapter.film.weight, std=0.5)
    hidden_states = [torch.randn(2, 6, 8) for _ in range(3)]

    first = torch.randn(2, 6, 4)
    second = torch.randn(2, 6, 4) * 17.0 + 3.0
    assert not torch.allclose(first, second)

    output_first = fusion(hidden_states, first, null_features=True)
    output_second = fusion(hidden_states, second, null_features=True)
    torch.testing.assert_close(output_first, output_second, atol=0, rtol=0)

    # With features enabled, the two representations should produce different outputs.
    live_first = fusion(hidden_states, first)
    live_second = fusion(hidden_states, second)
    assert not torch.allclose(live_first, live_second)


def test_ab3_encoder_weights_do_not_reach_the_fusion_input():
    """Two differently-trained encoders, same AB3 fusion input, to the bit."""
    from prosody_adaptation.fusion import LayerwisePostHocFusion

    torch.manual_seed(6)
    fusion = LayerwisePostHocFusion(hidden_dim=8, prosody_dim=8, layers=2).eval()
    hidden_states = [torch.randn(1, 5, 8) for _ in range(3)]
    waveform, lengths = torch.randn(1, SAMPLE_RATE), torch.tensor([SAMPLE_RATE])

    outputs = []
    for seed in (11, 22):
        torch.manual_seed(seed)
        encoder = ProsodyEncoderWithHeads(
            channels=16, gru_hidden=8, representation_dim=8
        ).encoder.eval()
        with torch.no_grad():
            prosody, _ = encoder(waveform, lengths)
            prosody = torch.nn.functional.interpolate(
                prosody.transpose(1, 2), size=5, mode="linear", align_corners=False
            ).transpose(1, 2)
            outputs.append(fusion(hidden_states, prosody, null_features=True))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
