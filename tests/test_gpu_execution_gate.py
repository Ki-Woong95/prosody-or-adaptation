import pytest
import torch

from prosody_adaptation.asr_model import PackedBLSTMHead, frozen_fp32_forward
from prosody_adaptation.checkpointing import restore_training_checkpoint, save_training_checkpoint

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution gate")


@CUDA_REQUIRED
def test_gpu_frozen_prosody_forward_stays_fp32_under_amp():
    class PrecisionProbe(torch.nn.Module):
        def forward(self, values, lengths):
            assert values.dtype == torch.float32
            assert not torch.is_autocast_enabled("cuda")
            return values.unsqueeze(-1), lengths

    values = torch.ones(2, 4, device="cuda")
    lengths = torch.tensor([4, 3], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, observed_lengths = frozen_fp32_forward(PrecisionProbe(), values, lengths)
    assert output.dtype == torch.float32
    torch.testing.assert_close(observed_lengths, lengths)


@CUDA_REQUIRED
def test_gpu_padding_invariance():
    torch.manual_seed(5)
    model = PackedBLSTMHead(4, 3, 1, 6, dropout=0).cuda().eval()
    values = torch.randn(2, 7, 4, device="cuda")
    lengths = torch.tensor([7, 4], device="cuda")
    extended = torch.nn.functional.pad(values, (0, 0, 0, 5))
    with torch.no_grad():
        original = model(values, lengths)
        padded = model(extended, lengths)
    torch.testing.assert_close(original[0, :7], padded[0, :7])
    torch.testing.assert_close(original[1, :4], padded[1, :4])


@CUDA_REQUIRED
def test_gpu_checkpoint_resume_preserves_step_and_rng(tmp_path):
    torch.manual_seed(9)
    torch.cuda.manual_seed_all(9)
    model = torch.nn.Linear(3, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(2, 3, device="cuda")).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    state = {"epoch": 2, "step": 17}
    save_training_checkpoint(path, model, optimizer, None, None, state, None)
    expected = torch.rand(3, device="cuda")
    restored = torch.nn.Linear(3, 2).cuda()
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    restored_state, _ = restore_training_checkpoint(path, restored, restored_optimizer)
    observed = torch.rand(3, device="cuda")
    assert restored_state == state
    torch.testing.assert_close(expected, observed)
