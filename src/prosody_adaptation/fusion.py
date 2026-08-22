from __future__ import annotations

import torch
from torch import nn


class IdentityInitializedAdapter(nn.Module):
    """Post-hoc hidden-state modulation whose residual branch is exactly zero at init."""

    def __init__(self, hidden_dim: int = 768, prosody_dim: int = 64):
        super().__init__()
        self.prosody_norm = nn.LayerNorm(prosody_dim)
        self.film = nn.Linear(prosody_dim, hidden_dim * 2)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim + prosody_dim, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, prosody, return_analysis: bool = False):
        normalized = self.prosody_norm(prosody)
        gamma, beta = self.film(normalized).chunk(2, dim=-1)
        candidate = self.candidate_norm(hidden + gamma * hidden + beta)
        gate = torch.sigmoid(self.gate(torch.cat((hidden, prosody), dim=-1)))
        contribution = torch.tanh(self.residual_scale) * gate * (candidate - hidden)
        output = hidden + contribution
        if return_analysis:
            return output, {"gate": gate, "residual_contribution": contribution}
        return output


class LayerwisePostHocFusion(nn.Module):
    def __init__(self, hidden_dim: int = 768, prosody_dim: int = 64, layers: int = 12):
        super().__init__()
        self.adapters = nn.ModuleList(
            IdentityInitializedAdapter(hidden_dim, prosody_dim) for _ in range(layers)
        )
        self.layer_logits = nn.Parameter(torch.zeros(layers + 1))

    def forward(self, hidden_states, prosody, null_features: bool = False, return_analysis=False):
        if len(hidden_states) != len(self.adapters) + 1:
            raise ValueError("Expected embedding output plus one state per transformer layer")
        if null_features:
            prosody = torch.zeros_like(prosody)
        modified = [hidden_states[0]]
        analysis = []
        for index, adapter in enumerate(self.adapters, start=1):
            if return_analysis:
                output, layer_analysis = adapter(
                    hidden_states[index], prosody, return_analysis=True
                )
                analysis.append(layer_analysis)
            else:
                output = adapter(hidden_states[index], prosody)
            modified.append(output)
        weights = self.layer_logits.softmax(dim=0)
        fused = sum(weight * state for weight, state in zip(weights, modified))
        return (fused, analysis) if return_analysis else fused
