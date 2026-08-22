import numpy as np
import yaml

from prosody_adaptation.teacher import (
    ENERGY_PERCENTILE,
    F0_MAX_HZ,
    F0_MIN_HZ,
    HOP_LENGTH,
    PERIODICITY_THRESHOLD,
    SAMPLE_RATE,
    WINDOW_LENGTH,
    _pitch_targets,
)


def test_teacher_voicing_and_delta_masks():
    frequency = np.array([100.0, 110.0, 120.0, 130.0, 600.0], dtype=np.float32)
    periodicity = np.array([0.02, 0.02, 0.001, 0.02, 0.02], dtype=np.float32)
    log_energy = np.array([-5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    log_f0, voiced, delta = _pitch_targets(frequency, periodicity, log_energy)

    np.testing.assert_array_equal(voiced, [0.0, 1.0, 0.0, 1.0, 0.0])
    assert log_f0[0] == 0.0
    assert log_f0[1] > 0.0
    np.testing.assert_array_equal(delta, np.zeros(5, dtype=np.float32))


def test_teacher_config_matches_implementation():
    with open("configs/teacher/casper.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["sample_rate"] == SAMPLE_RATE
    assert config["hop_length"] == HOP_LENGTH
    assert config["window_length"] == WINDOW_LENGTH
    assert config["energy_percentile"] == ENERGY_PERCENTILE
    assert config["periodicity"]["threshold"] == PERIODICITY_THRESHOLD
    assert config["crepe"]["f0_min_hz"] == F0_MIN_HZ
    assert config["crepe"]["f0_max_hz"] == F0_MAX_HZ
