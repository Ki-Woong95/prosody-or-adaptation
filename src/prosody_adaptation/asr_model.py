from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .ctc import LengthAwareCTC, greedy_decode
from .fusion import LayerwisePostHocFusion
from .lengths import frame_mask
from .prosody_model import ProsodyEncoderWithHeads


def frozen_fp32_forward(module, values, lengths):
    """Run a frozen acoustic frontend in FP32 even inside an AMP region."""
    with torch.autocast(device_type=values.device.type, enabled=False):
        return module(values.float(), lengths)


def align_prosody_to_frames(prosody, prosody_lengths, frame_lengths, width):
    """Resample valid prosody frames to each utterance's HuBERT frame length."""
    if prosody.shape[0] != len(prosody_lengths) or prosody.shape[0] != len(frame_lengths):
        raise ValueError("Prosody, prosody lengths and frame lengths must agree on batch size")
    aligned = prosody.new_zeros(prosody.shape[0], width, prosody.shape[-1])
    for index, (source, target) in enumerate(
        zip(prosody_lengths.tolist(), frame_lengths.tolist())
    ):
        if source <= 0 or target <= 0:
            raise ValueError(
                f"Utterance {index} has {source} prosody frames and {target} output frames; "
                "both must be positive"
            )
        if target > width:
            raise ValueError(
                f"Utterance {index} needs {target} output frames but the batch is {width} wide"
            )
        aligned[index, :target] = F.interpolate(
            prosody[index, :source].transpose(0, 1).unsqueeze(0),
            size=target, mode="linear", align_corners=False,
        ).squeeze(0).transpose(0, 1)
    return aligned


def intervene_prosody(prosody, frame_lengths, mode, donor=None, generator=None):
    if mode == "true":
        return prosody
    if mode == "zero":
        return torch.zeros_like(prosody)
    if mode == "utterance_shuffle":
        if donor is None or donor.shape != prosody.shape:
            raise ValueError("utterance_shuffle requires aligned donor features")
        return donor
    if mode != "time_shuffle":
        raise ValueError(f"Unsupported feature intervention: {mode}")
    shuffled = prosody.clone()
    for index, length in enumerate(frame_lengths.tolist()):
        order = torch.randperm(length, device=prosody.device, generator=generator)
        shuffled[index, :length] = prosody[index, order]
    return shuffled


class PackedBLSTMHead(nn.Module):
    def __init__(self, input_dim, hidden_size, layers, vocabulary_size, dropout=0.1, blank_id=0):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_dim, hidden_size, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size * 2, vocabulary_size)
        self.blank_id = blank_id

    def forward(self, values, lengths):
        packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.recurrent(packed)
        values, _ = pad_packed_sequence(packed, batch_first=True, total_length=values.shape[1])
        return self.projection(values)


class ProductionASR(nn.Module):
    """Frozen HuBERT with plain aggregation or post-hoc layer-wise modulation."""

    def __init__(
        self, config, vocabulary_size, condition, prosody_checkpoint=None, blank_id=0,
        local_files_only=False,
    ):
        super().__init__()
        if condition not in {"ab1", "ab3", "full"}:
            raise ValueError("Production backend currently supports AB1, AB3, and Full")
        from transformers import HubertModel

        self.condition = condition
        self.hubert = HubertModel.from_pretrained(
            config["hubert_model"], revision=config["hubert_revision"],
            local_files_only=local_files_only,
        )
        self.hubert.requires_grad_(False).eval()
        self.layer_logits = nn.Parameter(torch.zeros(13)) if condition == "ab1" else None
        self.fusion = (
            LayerwisePostHocFusion(config["hidden_dim"], config["prosody_dim"], 12)
            if condition != "ab1" else None
        )
        self.prosody_encoder = None
        if condition != "ab1":
            if not prosody_checkpoint:
                raise ValueError(f"{condition} requires the single frozen ProsodyEncoder checkpoint")
            payload = torch.load(prosody_checkpoint, map_location="cpu", weights_only=False)
            wrapper = ProsodyEncoderWithHeads(representation_dim=config["prosody_dim"])
            wrapper.load_state_dict(payload["model"])
            self.prosody_encoder = wrapper.encoder.requires_grad_(False).eval()
        self.projector = nn.Linear(config["hidden_dim"], config["project_dim"])
        self.head = PackedBLSTMHead(
            config["project_dim"], config["blstm_hidden"], config["blstm_layers"],
            vocabulary_size, config["dropout"], blank_id,
        )
        self.ctc = LengthAwareCTC(blank_id)

    def train(self, mode=True):
        super().train(mode)
        self.hubert.eval()
        if self.prosody_encoder is not None:
            self.prosody_encoder.eval()
        return self

    def _aggregate_ab1(self, hidden_states):
        weights = self.layer_logits.softmax(0)
        return sum(weight * state for weight, state in zip(weights, hidden_states))

    def forward(
        self, input_values, attention_mask, waveform_lengths, targets=None,
        target_lengths=None, return_analysis=False, feature_mode="true",
        donor_input_values=None, donor_waveform_lengths=None, generator=None,
    ):
        with torch.no_grad():
            hidden_states = self.hubert(
                input_values, attention_mask=attention_mask, output_hidden_states=True,
                return_dict=True,
            ).hidden_states
        frame_lengths = self.hubert._get_feat_extract_output_lengths(waveform_lengths)
        analysis = None
        if self.condition == "ab1":
            fused = self._aggregate_ab1(hidden_states)
        else:
            width = hidden_states[0].shape[1]
            with torch.no_grad():
                prosody, prosody_lengths = frozen_fp32_forward(
                    self.prosody_encoder, input_values, waveform_lengths
                )
            if not torch.isfinite(prosody).all():
                raise FloatingPointError("Non-finite prosody features")
            prosody = align_prosody_to_frames(
                prosody, prosody_lengths, frame_lengths, width
            ).to(dtype=hidden_states[0].dtype)
            donor = None
            if feature_mode == "utterance_shuffle":
                if donor_input_values is None or donor_waveform_lengths is None:
                    raise ValueError("utterance_shuffle requires donor waveforms")
                with torch.no_grad():
                    donor, donor_lengths = frozen_fp32_forward(
                        self.prosody_encoder, donor_input_values, donor_waveform_lengths
                    )
                # Align the donor contour to the recipient frame count.
                donor = align_prosody_to_frames(
                    donor, donor_lengths, frame_lengths, width
                ).to(dtype=hidden_states[0].dtype)
            prosody = intervene_prosody(
                prosody, frame_lengths, feature_mode, donor=donor, generator=generator
            )
            fusion_output = self.fusion(
                hidden_states, prosody, null_features=self.condition == "ab3",
                return_analysis=return_analysis,
            )
            fused, analysis = fusion_output if return_analysis else (fusion_output, None)
        logits = self.head(self.projector(fused), frame_lengths)
        output = {"logits": logits, "frame_lengths": frame_lengths}
        if analysis:
            valid = frame_mask(frame_lengths, fused.shape[1])
            gates = torch.stack([item["gate"].squeeze(-1) for item in analysis])
            residual_norms = torch.stack([
                item["residual_contribution"].float().norm(dim=-1) for item in analysis
            ])
            layer_valid = valid.unsqueeze(0).expand_as(gates)
            output["mean_gate_activation"] = gates[layer_valid].mean()
            output["mean_residual_norm"] = residual_norms[layer_valid].mean()
            counts = valid.sum().expand(len(analysis))
            output["layer_gate_sum"] = (gates * layer_valid).sum(dim=(1, 2))
            output["layer_residual_norm_sum"] = (
                residual_norms * layer_valid
            ).sum(dim=(1, 2))
            output["layer_valid_frame_count"] = counts
        if targets is not None:
            output["loss"], observed = self.ctc(logits, targets, waveform_lengths, target_lengths)
            torch.testing.assert_close(observed, frame_lengths)
        return output

    def decode(self, output):
        return greedy_decode(output["logits"], output["frame_lengths"], self.head.blank_id)

    def parameter_counts(self):
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters()),
            "hubert_frozen": not any(p.requires_grad for p in self.hubert.parameters()),
            "prosody_encoder_frozen": self.prosody_encoder is None or not any(
                p.requires_grad for p in self.prosody_encoder.parameters()
            ),
        }
