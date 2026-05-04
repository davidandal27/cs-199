from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F


@contextmanager
def freeze_model_parameters(model):
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(flag)


def generate_fgsm_batch(
    model,
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
        raise ValueError(
            f"Invalid clamp range: clamp_min ({clamp_min}) > clamp_max ({clamp_max})."
        )
    if not waveforms.is_floating_point():
        raise TypeError("FGSM expects floating-point waveform tensors.")

    model.eval()
    labels = labels.view(-1).to(dtype=torch.int64, device=waveforms.device)
    clean_waveforms = waveforms.detach()
    adv_waveforms = clean_waveforms.clone()
    adv_waveforms.requires_grad_(True)

    with torch.enable_grad():
        with freeze_model_parameters(model):
            model.zero_grad(set_to_none=True)
            logits = model(adv_waveforms)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

    gradients = adv_waveforms.grad
    if gradients is None:
        raise RuntimeError("FGSM attack expected input gradients, but none were produced.")

    perturbation = epsilon * gradients.sign()
    adversarial = adv_waveforms + perturbation
    if clamp_min is not None or clamp_max is not None:
        adversarial = adversarial.clamp(min=clamp_min, max=clamp_max)
    adversarial = adversarial.detach()

    stats = {}
    if return_stats:
        applied_perturbation = adversarial - clean_waveforms
        flattened = applied_perturbation.abs().view(applied_perturbation.size(0), -1)
        stats = {
            "loss": float(loss.detach().item()),
            "epsilon": float(epsilon),
            "max_abs_perturbation": float(flattened.max().item()),
            "mean_abs_perturbation": float(flattened.mean().item()),
            "mean_l2_perturbation": float(
                applied_perturbation.view(applied_perturbation.size(0), -1)
                .norm(p=2, dim=1)
                .mean()
                .item()
            ),
        }

    model.zero_grad(set_to_none=True)
    return adversarial.to(dtype=waveforms.dtype), stats


def save_adversarial_waveforms(
    waveforms: torch.Tensor,
    utterance_ids: Sequence[str],
    output_dir: Path,
    sample_rate: int = 16000,
) -> Iterable[Path]:
    if waveforms.size(0) != len(utterance_ids):
        raise ValueError(
            "Waveform batch size and utterance ID count must match when saving audio."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for waveform, utterance_id in zip(waveforms.detach().cpu(), utterance_ids):
        save_path = output_dir / f"{utterance_id}.wav"
        sf.write(save_path, waveform.numpy(), sample_rate)
        saved_paths.append(save_path)
    return saved_paths
