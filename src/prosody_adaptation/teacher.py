from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from .lengths import prosody_frame_count

SAMPLE_RATE = 16_000
HOP_LENGTH = 320
WINDOW_LENGTH = 800
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0
ENERGY_PERCENTILE = 20
PERIODICITY_THRESHOLD = 0.01

# torchcrepe pads by WINDOW_SIZE // 2 and unfolds with stride=hop, so CREPE frame
# i is centred at sample i * HOP_LENGTH. The canonical (student) grid centres
# frame j at j * HOP_LENGTH + WINDOW_LENGTH / 2. Canonical frame j therefore sits
# at CREPE position j + CREPE_FRAME_OFFSET.
CREPE_FRAME_OFFSET = WINDOW_LENGTH / (2 * HOP_LENGTH)  # 1.25 frames = 25 ms


@dataclass(frozen=True)
class TeacherTargets:
    log_f0: np.ndarray
    voiced: np.ndarray
    periodicity: np.ndarray
    delta_log_f0: np.ndarray
    log_energy: np.ndarray
    spectral_tilt: np.ndarray

    def as_columns(self) -> dict[str, list[float]]:
        return {
            "teacher_log_f0": self.log_f0.tolist(),
            "teacher_crepe_conf": self.voiced.tolist(),
            "teacher_crepe_periodicity": self.periodicity.tolist(),
            "teacher_delta_log_f0": self.delta_log_f0.tolist(),
            "teacher_log_energy": self.log_energy.tolist(),
            "teacher_spectral_tilt": self.spectral_tilt.tolist(),
        }


def _to_canonical_grid(values: np.ndarray, frames: int) -> np.ndarray:
    """Resample a CREPE-grid signal onto the canonical frame centers."""
    if len(values) == 0:
        return np.zeros(frames, dtype=np.float32)
    query = np.arange(frames, dtype=np.float64) + CREPE_FRAME_OFFSET
    source = np.arange(len(values), dtype=np.float64)
    return np.interp(query, source, values.astype(np.float64)).astype(np.float32)


def _median_filter(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (1, 1), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + 3]) for index in range(len(values))],
        dtype=np.float32,
    )


def _pitch_targets(
    frequency: np.ndarray,
    periodicity: np.ndarray,
    log_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voiced = (
        (log_energy > np.percentile(log_energy, ENERGY_PERCENTILE))
        & np.isfinite(frequency)
        & (frequency >= F0_MIN_HZ)
        & (frequency <= F0_MAX_HZ)
        & (periodicity >= PERIODICITY_THRESHOLD)
    )
    log_f0 = np.zeros(len(frequency), dtype=np.float32)
    log_f0[voiced] = np.log(np.maximum(frequency[voiced], 1.0)).astype(np.float32)

    transitions = np.zeros(len(voiced), dtype=bool)
    transitions[1:] = voiced[1:] & voiced[:-1]
    delta = np.zeros(len(frequency), dtype=np.float32)
    delta[1:] = log_f0[1:] - log_f0[:-1]
    delta[~transitions] = 0.0
    return log_f0, voiced.astype(np.float32), delta


class TeacherExtractor:
    def __init__(self, crepe_model: str = "tiny", device: str = "cuda"):
        self.crepe_model = crepe_model
        self.device = device
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WINDOW_LENGTH,
            win_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=80,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            # center=False places frame j at [j*hop, j*hop + win), which is the
            # canonical grid the student's LogMelFrontend produces. The framing is
            # bitwise identical to the student's transform.
            center=False,
            power=2.0,
        )

    def extract(self, waveform: np.ndarray) -> TeacherTargets:
        audio = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        expected_frames = prosody_frame_count(len(audio))
        with torch.no_grad():
            log_mel = torch.log(self.mel(audio[None]) + 1e-6)[0]
        if log_mel.shape[1] != expected_frames:
            raise ValueError(
                f"Teacher mel produced {log_mel.shape[1]} frames but the canonical grid "
                f"defines {expected_frames} for {len(audio)} samples"
            )
        log_energy = log_mel.mean(0).cpu().numpy().astype(np.float32)
        midpoint = log_mel.shape[0] // 2
        spectral_tilt = (
            log_mel[midpoint:].mean(0) - log_mel[:midpoint].mean(0)
        ).cpu().numpy().astype(np.float32)

        frequency, periodicity = self._crepe(audio, len(log_energy))
        log_f0, voiced, delta = _pitch_targets(frequency, periodicity, log_energy)
        return TeacherTargets(
            log_f0,
            voiced,
            periodicity,
            delta,
            log_energy,
            spectral_tilt,
        )

    def _crepe(self, waveform: torch.Tensor, frames: int) -> tuple[np.ndarray, np.ndarray]:
        import torchcrepe

        device = torch.device(self.device)
        with torch.no_grad():
            frequency, periodicity = torchcrepe.predict(
                waveform[None].to(device),
                SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                fmin=F0_MIN_HZ,
                fmax=F0_MAX_HZ,
                model=self.crepe_model,
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                device=device,
                batch_size=512,
            )
        # Smooth periodicity on CREPE's native grid before timestamp alignment.
        frequency = _to_canonical_grid(frequency[0].cpu().numpy(), frames)
        periodicity = _to_canonical_grid(_median_filter(periodicity[0].cpu().numpy()), frames)
        return frequency, periodicity
