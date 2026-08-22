from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .asr_data import ASRCollator, load_asr_splits
from .asr_model import (
    ProductionASR,
    align_prosody_to_frames,
    frozen_fp32_forward,
    intervene_prosody,
)
from .asr_train import _edit_distance, _move
from .lengths import frame_mask


MODES = ("true", "zero", "time_shuffle", "utterance_shuffle")


def record_path(path, root):
    """Return a repo-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        return str(resolved)


class ShiftedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.offset = max(1, len(dataset) // 2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[(index + self.offset) % len(self.dataset)]


def _accumulate_layer_analysis(totals, analysis, hidden_states, valid):
    gates = torch.stack([entry["gate"].squeeze(-1) for entry in analysis])
    residuals = torch.stack([
        entry["residual_contribution"].float().norm(dim=-1) for entry in analysis
    ])
    hidden_norms = torch.stack([
        state.float().norm(dim=-1) for state in hidden_states[1:]
    ])
    relative = residuals / hidden_norms.clamp_min(1e-12)
    layer_valid = valid.unsqueeze(0).expand_as(gates)
    values = {
        "gate": (gates * layer_valid).sum(dim=(1, 2)).double().cpu(),
        "residual": (residuals * layer_valid).sum(dim=(1, 2)).double().cpu(),
        "hidden": (hidden_norms * layer_valid).sum(dim=(1, 2)).double().cpu(),
        "relative_residual": (
            relative * layer_valid
        ).sum(dim=(1, 2)).double().cpu(),
        "counts": valid.sum().expand(len(analysis)).double().cpu(),
    }
    for key, value in values.items():
        totals[key] = value if totals[key] is None else totals[key] + value


def _finish_layer_analysis(totals):
    counts = totals["counts"]
    return {
        "valid_frames": int(counts[0].item()),
        "layer_gate_mean": (totals["gate"] / counts).tolist(),
        "layer_residual_norm_mean": (totals["residual"] / counts).tolist(),
        "layer_hidden_norm_mean": (totals["hidden"] / counts).tolist(),
        "layer_relative_residual_mean": (
            totals["relative_residual"] / counts
        ).tolist(),
    }


@torch.no_grad()
def evaluate_residual_fusions(frontend_model, fusions, loader, device, max_batches=None):
    frontend_model.eval()
    for fusion in fusions.values():
        fusion.eval()
    totals = {
        name: {
            "gate": None, "residual": None, "hidden": None,
            "relative_residual": None, "counts": None,
        }
        for name in fusions
    }
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move(batch, device)
        hidden_states = frontend_model.hubert(
            batch["input_values"], attention_mask=batch["attention_mask"],
            output_hidden_states=True, return_dict=True,
        ).hidden_states
        frame_lengths = frontend_model.hubert._get_feat_extract_output_lengths(
            batch["waveform_lengths"]
        )
        width = hidden_states[0].shape[1]
        prosody, prosody_lengths = frozen_fp32_forward(
            frontend_model.prosody_encoder, batch["input_values"], batch["waveform_lengths"]
        )
        prosody = align_prosody_to_frames(
            prosody, prosody_lengths, frame_lengths, width
        ).to(dtype=hidden_states[0].dtype)
        valid = frame_mask(frame_lengths, width)
        for name, fusion in fusions.items():
            _, analysis = fusion(
                hidden_states,
                prosody,
                null_features=name.endswith(":ab3"),
                return_analysis=True,
            )
            _accumulate_layer_analysis(totals[name], analysis, hidden_states, valid)
    if next(iter(totals.values()))["counts"] is None:
        raise ValueError("Residual analysis evaluated no utterances")
    return {
        name: _finish_layer_analysis(values) for name, values in totals.items()
    }


@torch.no_grad()
def evaluate_interventions(
    model, loader, donor_loader, processor, device, max_batches=None, modes=MODES,
    utterance_sink=None,
):
    """Evaluate feature interventions and optionally record utterance-level errors."""
    model.eval()
    totals = {
        mode: {"errors": 0, "words": 0, "utterances": 0, "gate": None,
               "residual": None, "hidden": None, "relative_residual": None,
               "counts": None}
        for mode in modes
    }
    generator = torch.Generator(device=device).manual_seed(42)
    for batch_index, (batch, donor) in enumerate(zip(loader, donor_loader, strict=True)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move(batch, device)
        donor = _move(donor, device)
        hidden_states = model.hubert(
            batch["input_values"], attention_mask=batch["attention_mask"],
            output_hidden_states=True, return_dict=True,
        ).hidden_states
        frame_lengths = model.hubert._get_feat_extract_output_lengths(batch["waveform_lengths"])
        width = hidden_states[0].shape[1]
        prosody, prosody_lengths = frozen_fp32_forward(
            model.prosody_encoder, batch["input_values"], batch["waveform_lengths"]
        )
        donor_prosody, donor_lengths = frozen_fp32_forward(
            model.prosody_encoder, donor["input_values"], donor["waveform_lengths"]
        )

        def align(values, lengths):
            return align_prosody_to_frames(
                values, lengths, frame_lengths, width
            ).to(dtype=hidden_states[0].dtype)

        prosody = align(prosody, prosody_lengths)
        donor_prosody = align(donor_prosody, donor_lengths)
        valid = frame_mask(frame_lengths, width)
        for mode in modes:
            features = intervene_prosody(
                prosody, frame_lengths, mode, donor=donor_prosody, generator=generator
            )
            fused, analysis = model.fusion(
                hidden_states, features, null_features=model.condition == "ab3",
                return_analysis=True,
            )
            logits = model.head(model.projector(fused), frame_lengths)
            decoded = model.decode({"logits": logits, "frame_lengths": frame_lengths})
            item = totals[mode]
            for index, token_ids in enumerate(decoded):
                hypothesis = processor.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).lower().strip()
                reference = batch["text"][index]
                word_errors = _edit_distance(reference.split(), hypothesis.split())
                reference_words = max(len(reference.split()), 1)
                item["errors"] += word_errors
                item["words"] += reference_words
                item["utterances"] += 1
                if utterance_sink is not None:
                    utterance_sink(mode, {
                        "segment_id": batch["segment_id"][index],
                        "speaker_id": batch["speaker_id"][index],
                        "word_errors": int(word_errors),
                        "reference_words": int(reference_words),
                        "reference": reference,
                    })
            gates = torch.stack([entry["gate"].squeeze(-1) for entry in analysis])
            residuals = torch.stack([
                entry["residual_contribution"].float().norm(dim=-1) for entry in analysis
            ])
            hidden_norms = torch.stack([
                state.float().norm(dim=-1) for state in hidden_states[1:]
            ])
            relative_residuals = residuals / hidden_norms.clamp_min(1e-12)
            layer_valid = valid.unsqueeze(0).expand_as(gates)
            current = (
                (gates * layer_valid).sum(dim=(1, 2)).double().cpu(),
                (residuals * layer_valid).sum(dim=(1, 2)).double().cpu(),
                (hidden_norms * layer_valid).sum(dim=(1, 2)).double().cpu(),
                (relative_residuals * layer_valid).sum(dim=(1, 2)).double().cpu(),
                valid.sum().expand(len(analysis)).double().cpu(),
            )
            for key, value in zip(
                ("gate", "residual", "hidden", "relative_residual", "counts"),
                current,
                strict=True,
            ):
                item[key] = value if item[key] is None else item[key] + value
    results = []
    for mode, item in totals.items():
        if not item["utterances"]:
            raise ValueError("Intervention analysis evaluated no utterances")
        results.append({
            "mode": mode, "wer": item["errors"] / item["words"],
            "word_errors": item["errors"], "reference_words": item["words"],
            "utterances": item["utterances"],
            "layer_gate_mean": (item["gate"] / item["counts"]).tolist(),
            "layer_residual_norm_mean": (item["residual"] / item["counts"]).tolist(),
            "layer_hidden_norm_mean": (item["hidden"] / item["counts"]).tolist(),
            "layer_relative_residual_mean": (
                item["relative_residual"] / item["counts"]
            ).tolist(),
        })
    return results


def run_feature_analysis(
    registry_path, output_path, device="cuda", max_batches=None, corpus_filter=None,
    seeds=(1, 2, 3), batch_size=None,
):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --device cpu only for a smoke test")
    torch_device = torch.device(device)
    root = Path(registry_path).resolve().parents[2]
    registry = yaml.safe_load(Path(registry_path).read_text())
    results = {
        "modes": list(MODES),
        "random_seed": 42,
        "utterance_shuffle": "fixed half-dataset circular shift",
        "max_batches": max_batches,
        "corpora": {},
    }
    from transformers import Wav2Vec2Processor

    # Keep per-utterance counts for paired intervention inference.
    utterance_dir = (Path(output_path).parent / "interventions").resolve()
    utterance_dir.mkdir(parents=True, exist_ok=True)
    results["utterance_record_directory"] = record_path(utterance_dir, root)

    for corpus, patterns in registry["corpora"].items():
        if corpus_filter and corpus != corpus_filter:
            continue
        corpus_results = {}
        for seed in seeds:
            run = root / patterns["full"].format(seed=seed)
            resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())
            experiment, model_config, base = (
                resolved["experiment"], resolved["model"], resolved["base"]
            )
            processor = Wav2Vec2Processor.from_pretrained(
                base["processor_model"], revision=base["processor_revision"],
                local_files_only=True,
            )
            test = load_asr_splits(experiment)[2]
            options = {
                "batch_size": batch_size or experiment["batch_size"],
                "shuffle": False,
                "num_workers": 0,
                "collate_fn": ASRCollator(processor),
            }
            loader = DataLoader(test, **options)
            donor_loader = DataLoader(ShiftedDataset(test), **options)
            model = ProductionASR(
                base,
                processor.tokenizer.vocab_size,
                model_config["condition"],
                experiment["prosody_checkpoint"],
                processor.tokenizer.pad_token_id,
                local_files_only=True,
            ).to(torch_device)
            payload = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            handles, written = {}, {}
            for mode in MODES:
                path = utterance_dir / f"{corpus}_seed{seed}_{mode}.jsonl"
                handles[mode] = path.open("w", encoding="utf-8")
                written[mode] = record_path(path, root)

            def sink(mode, row, handles=handles):
                handles[mode].write(json.dumps(row, sort_keys=True) + "\n")

            try:
                mode_results = evaluate_interventions(
                    model, loader, donor_loader, processor, torch_device, max_batches,
                    utterance_sink=sink,
                )
            finally:
                for handle in handles.values():
                    handle.close()
            seed_results = {
                "source_run": str(run.relative_to(root)),
                "checkpoint_sha256": (run / "checkpoint_best_sha256.txt").read_text().strip(),
                "results": mode_results,
                "utterance_records": written,
            }
            del model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            ab3_run = root / patterns["ab3"].format(seed=seed)
            ab3_model = ProductionASR(
                base,
                processor.tokenizer.vocab_size,
                "ab3",
                experiment["prosody_checkpoint"],
                processor.tokenizer.pad_token_id,
                local_files_only=True,
            ).to(torch_device)
            payload = torch.load(
                ab3_run / "checkpoint_best.pt", map_location="cpu", weights_only=False
            )
            ab3_model.load_state_dict(payload["model"])
            seed_results["ab3_analysis"] = {
                "source_run": str(ab3_run.relative_to(root)),
                "checkpoint_sha256": (
                    ab3_run / "checkpoint_best_sha256.txt"
                ).read_text().strip(),
                "results": evaluate_interventions(
                    ab3_model,
                    loader,
                    donor_loader,
                    processor,
                    torch_device,
                    max_batches,
                    modes=("true",),
                ),
            }
            corpus_results[str(seed)] = seed_results
            del ab3_model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        results["corpora"][corpus] = corpus_results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def run_residual_analysis(
    registry_path, output_path, device="cuda", max_batches=None, corpus_filter=None,
    seeds=(1, 2, 3), batch_size=None,
):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --device cpu only for a smoke test")
    torch_device = torch.device(device)
    root = Path(registry_path).resolve().parents[2]
    registry = yaml.safe_load(Path(registry_path).read_text())
    output = {
        "definition": "mean over valid frames of L2 norms; relative = ||r_l||_2 / ||h_l||_2",
        "max_batches": max_batches,
        "corpora": {},
    }
    from transformers import Wav2Vec2Processor

    for corpus, patterns in registry["corpora"].items():
        if corpus_filter and corpus != corpus_filter:
            continue
        corpus_results = {}
        first_seed = seeds[0]
        first_run = root / patterns["full"].format(seed=first_seed)
        resolved = yaml.safe_load((first_run / "config.resolved.yaml").read_text())
        experiment, model_config, base = (
            resolved["experiment"], resolved["model"], resolved["base"]
        )
        processor = Wav2Vec2Processor.from_pretrained(
            base["processor_model"], revision=base["processor_revision"],
            local_files_only=True,
        )
        test = load_asr_splits(experiment)[2]
        loader = DataLoader(
            test,
            batch_size=batch_size or experiment["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=ASRCollator(processor),
        )
        frontend_model = ProductionASR(
            base,
            processor.tokenizer.vocab_size,
            model_config["condition"],
            experiment["prosody_checkpoint"],
            processor.tokenizer.pad_token_id,
            local_files_only=True,
        )
        first_payload = torch.load(
            first_run / "checkpoint_best.pt", map_location="cpu", weights_only=False
        )
        frontend_model.load_state_dict(first_payload["model"])
        frontend_model.to(torch_device)
        fusions = {f"{first_seed}:full": frontend_model.fusion}
        for seed in seeds:
            full_run = root / patterns["full"].format(seed=seed)
            ab3_run = root / patterns["ab3"].format(seed=seed)
            if seed != first_seed:
                model = ProductionASR(
                    base,
                    processor.tokenizer.vocab_size,
                    "full",
                    experiment["prosody_checkpoint"],
                    processor.tokenizer.pad_token_id,
                    local_files_only=True,
                )
                payload = torch.load(
                    full_run / "checkpoint_best.pt", map_location="cpu", weights_only=False
                )
                model.load_state_dict(payload["model"])
                fusions[f"{seed}:full"] = model.fusion.to(torch_device)
                del model
            model = ProductionASR(
                base,
                processor.tokenizer.vocab_size,
                "ab3",
                experiment["prosody_checkpoint"],
                processor.tokenizer.pad_token_id,
                local_files_only=True,
            )
            payload = torch.load(
                ab3_run / "checkpoint_best.pt", map_location="cpu", weights_only=False
            )
            model.load_state_dict(payload["model"])
            fusions[f"{seed}:ab3"] = model.fusion.to(torch_device)
            del model
            corpus_results[str(seed)] = {
                "full_source_run": str(full_run.relative_to(root)),
                "full_checkpoint_sha256": (
                    full_run / "checkpoint_best_sha256.txt"
                ).read_text().strip(),
                "ab3_source_run": str(ab3_run.relative_to(root)),
                "ab3_checkpoint_sha256": (
                    ab3_run / "checkpoint_best_sha256.txt"
                ).read_text().strip(),
            }
        statistics = evaluate_residual_fusions(
            frontend_model, fusions, loader, torch_device, max_batches
        )
        for seed in seeds:
            corpus_results[str(seed)]["conditions"] = {
                condition: statistics[f"{seed}:{condition}"]
                for condition in ("full", "ab3")
            }
            # softmax over the 13 aggregation weights (embedding output + 12 layers)
            for condition in ("full", "ab3"):
                fusion = fusions[f"{seed}:{condition}"]
                corpus_results[str(seed)]["conditions"][condition][
                    "layer_aggregation_weights"
                ] = torch.softmax(
                    fusion.layer_logits.detach().float(), dim=0
                ).cpu().tolist()
        del frontend_model, fusions
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        output["corpora"][corpus] = corpus_results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output
