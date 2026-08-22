import torch
from torch import nn

from prosody_adaptation.asr_model import frozen_fp32_forward, intervene_prosody
from prosody_adaptation.fusion import IdentityInitializedAdapter


def test_identity_initialization():
    torch.manual_seed(1)
    module = IdentityInitializedAdapter(hidden_dim=8, prosody_dim=3)
    hidden = torch.randn(2, 5, 8)
    prosody = torch.randn(2, 5, 3)
    output = module(hidden, prosody)
    torch.testing.assert_close(output, hidden, atol=0, rtol=0)


def test_ab3_feature_independence_when_features_are_zeroed():
    module = IdentityInitializedAdapter(hidden_dim=8, prosody_dim=3)
    with torch.no_grad():
        module.residual_scale.fill_(0.5)
    hidden = torch.randn(2, 5, 8)
    zero = torch.zeros(2, 5, 3)
    output_a = module(hidden, zero)
    output_b = module(hidden, torch.zeros_like(zero))
    torch.testing.assert_close(output_a, output_b)


def test_checkpoint_save_load_equivalence(tmp_path):
    module = IdentityInitializedAdapter(hidden_dim=8, prosody_dim=3)
    path = tmp_path / "checkpoint.pt"
    torch.save(module.state_dict(), path)
    restored = IdentityInitializedAdapter(hidden_dim=8, prosody_dim=3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    hidden, prosody = torch.randn(2, 4, 8), torch.randn(2, 4, 3)
    torch.testing.assert_close(module(hidden, prosody), restored(hidden, prosody))


def test_frozen_prosody_forward_disables_outer_autocast():
    class PrecisionProbe(nn.Module):
        def forward(self, values, lengths):
            assert values.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            return values.unsqueeze(-1), lengths

    values = torch.ones(2, 4, dtype=torch.bfloat16)
    lengths = torch.tensor([4, 3])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, observed_lengths = frozen_fp32_forward(PrecisionProbe(), values, lengths)
    assert output.dtype == torch.float32
    torch.testing.assert_close(observed_lengths, lengths)


def test_feature_interventions_preserve_shape_and_padding():
    prosody = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    lengths = torch.tensor([4, 2])
    generator = torch.Generator().manual_seed(7)
    shuffled = intervene_prosody(prosody, lengths, "time_shuffle", generator=generator)
    assert shuffled.shape == prosody.shape
    torch.testing.assert_close(shuffled[1, 2:], prosody[1, 2:])
    assert sorted(shuffled[1, :2, 0].tolist()) == sorted(prosody[1, :2, 0].tolist())
    torch.testing.assert_close(intervene_prosody(prosody, lengths, "zero"), torch.zeros_like(prosody))
    donor = prosody.flip(0)
    torch.testing.assert_close(
        intervene_prosody(prosody, lengths, "utterance_shuffle", donor=donor), donor
    )
