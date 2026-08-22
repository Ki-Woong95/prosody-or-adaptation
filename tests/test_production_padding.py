import torch

from prosody_adaptation.asr_model import PackedBLSTMHead
from prosody_adaptation.prosody_model import ProsodyEncoderWithHeads


def test_prosody_frontend_and_encoder_padding_invariance():
    torch.manual_seed(11)
    model = ProsodyEncoderWithHeads(channels=16, gru_hidden=8, representation_dim=8).eval()
    waveform = torch.randn(1, 3200)
    padded = torch.cat((waveform, torch.zeros(1, 1600)), dim=1)
    first = model(waveform, torch.tensor([3200]))["representation"]
    second = model(padded, torch.tensor([3200]))["representation"]
    torch.testing.assert_close(first, second[:, : first.shape[1]], atol=1e-6, rtol=1e-6)


def test_downstream_blstm_padding_invariance():
    torch.manual_seed(12)
    model = PackedBLSTMHead(4, 3, 2, 5).eval()
    valid = torch.randn(1, 6, 4)
    padded = torch.cat((valid, torch.randn(1, 4, 4)), dim=1)
    first = model(valid, torch.tensor([6]))
    second = model(padded, torch.tensor([6]))
    torch.testing.assert_close(first, second[:, :6], atol=1e-6, rtol=1e-6)

