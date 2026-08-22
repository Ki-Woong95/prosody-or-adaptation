from __future__ import annotations

import torch
from torch import nn

from .prosody import PackedBiGRU


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, win_ms=50, hop_ms=20):
        super().__init__()
        try:
            import torchaudio
        except ImportError as error:
            raise ImportError("Install prosody-adaptation[train] for the log-mel frontend") from error
        self.sample_rate = sample_rate
        self.hop_length = round(sample_rate * hop_ms / 1000)
        win_length = round(sample_rate * win_ms / 1000)
        self.win_length = win_length
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=win_length, win_length=win_length,
            hop_length=self.hop_length, n_mels=n_mels, center=False, power=2.0,
        )

    def forward(self, waveforms, waveform_lengths):
        mel = self.transform(waveforms.float()).clamp_min(1e-6).log().transpose(1, 2)
        frame_lengths = (
            (waveform_lengths - self.win_length).div(self.hop_length, rounding_mode="floor") + 1
        ).clamp_min(1)
        frame_lengths = frame_lengths.clamp_max(mel.shape[1])
        return mel, frame_lengths


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, values):
        residual = values
        values = self.pointwise(self.depthwise(values)).transpose(1, 2)
        return residual + self.activation(self.norm(values)).transpose(1, 2)


class ProsodyEncoder(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, channels=128, gru_hidden=64, representation_dim=64):
        super().__init__()
        self.frontend = LogMelFrontend(sample_rate, n_mels)
        self.input_projection = nn.Sequential(nn.Linear(n_mels, channels), nn.LayerNorm(channels))
        self.convolution = nn.ModuleList(ConvBlock(channels, kernel) for kernel in (3, 5, 5, 9))
        self.recurrent = PackedBiGRU(channels, gru_hidden)
        self.output_projection = nn.Sequential(nn.Linear(gru_hidden * 2, representation_dim), nn.LayerNorm(representation_dim))

    def forward(self, waveforms, waveform_lengths):
        values, frame_lengths = self.frontend(waveforms, waveform_lengths)
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < frame_lengths[:, None]
        values = self.input_projection(values) * valid.unsqueeze(-1)
        values = values.transpose(1, 2)
        channel_mask = valid.unsqueeze(1)
        for block in self.convolution:
            values = block(values) * channel_mask
        values = values.transpose(1, 2)
        values = self.recurrent(values, frame_lengths)
        return self.output_projection(values), frame_lengths


class ProsodyEncoderWithHeads(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ProsodyEncoder(**kwargs)
        dim = kwargs.get("representation_dim", 64)
        self.heads = nn.ModuleDict({name: nn.Linear(dim, 1) for name in ("log_f0", "voicing_logits", "delta_f0", "energy", "tilt")})

    def forward(self, waveforms, waveform_lengths):
        representation, frame_lengths = self.encoder(waveforms, waveform_lengths)
        output = {name: head(representation).squeeze(-1) for name, head in self.heads.items()}
        output["representation"] = representation
        output["frame_lengths"] = frame_lengths
        return output
