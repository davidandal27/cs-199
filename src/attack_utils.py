from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F

from src.pgd_utils import DEFAULT_CLAMP_MAX, DEFAULT_CLAMP_MIN, PgdAttackConfig


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


def _validate_attack_inputs(
    attack_name: str,
    waveforms: torch.Tensor,
    epsilon: float,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> None:
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
        raise ValueError(
            f"Invalid clamp range: clamp_min ({clamp_min}) > clamp_max ({clamp_max})."
        )
    if not waveforms.is_floating_point():
        raise TypeError(f"{attack_name} expects floating-point waveform tensors.")


def _summarize_attack_stats(
    clean_waveforms: torch.Tensor,
    adversarial_waveforms: torch.Tensor,
    loss: torch.Tensor,
    extra_fields: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    applied_perturbation = adversarial_waveforms - clean_waveforms
    flattened = applied_perturbation.abs().reshape(applied_perturbation.size(0), -1)
    stats = {
        "loss": float(loss.detach().item()),
        "max_abs_perturbation": float(flattened.max().item()),
        "mean_abs_perturbation": float(flattened.mean().item()),
        "mean_l2_perturbation": float(
            applied_perturbation.reshape(applied_perturbation.size(0), -1)
            .norm(p=2, dim=1)
            .mean()
            .item()
        ),
    }
    if extra_fields:
        stats.update(extra_fields)
    return stats


def _clamp_waveforms(
    waveforms: torch.Tensor,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> torch.Tensor:
    if clamp_min is None and clamp_max is None:
        return waveforms
    return waveforms.clamp(min=clamp_min, max=clamp_max)


def generate_fgsm_batch(
    model,
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _validate_attack_inputs(
        attack_name="FGSM",
        waveforms=waveforms,
        epsilon=epsilon,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

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
    adversarial = _clamp_waveforms(
        waveforms=adversarial,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    adversarial = adversarial.detach()

    stats = {}
    if return_stats:
        stats = _summarize_attack_stats(
            clean_waveforms=clean_waveforms,
            adversarial_waveforms=adversarial,
            loss=loss,
            extra_fields={"epsilon": float(epsilon)},
        )

    model.zero_grad(set_to_none=True)
    return adversarial.to(dtype=waveforms.dtype), stats


def generate_pgd_batch(
    model,
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int = 5,
    alpha: Optional[float] = None,
    random_start: bool = False,
    clamp_min: float = DEFAULT_CLAMP_MIN,
    clamp_max: float = DEFAULT_CLAMP_MAX,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    attack_config = PgdAttackConfig(
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        random_start=random_start,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    _validate_attack_inputs(
        attack_name="PGD",
        waveforms=waveforms,
        epsilon=attack_config.epsilon,
        clamp_min=attack_config.clamp_min,
        clamp_max=attack_config.clamp_max,
    )

    model.eval()
    labels = labels.view(-1).to(dtype=torch.int64, device=waveforms.device)
    clean_waveforms = waveforms.detach()
    adversarial = clean_waveforms.clone()

    if attack_config.random_start and attack_config.epsilon > 0:
        random_delta = torch.empty_like(adversarial).uniform_(
            -attack_config.epsilon, attack_config.epsilon
        )
        adversarial = clean_waveforms + random_delta
        adversarial = _clamp_waveforms(
            waveforms=adversarial,
            clamp_min=attack_config.clamp_min,
            clamp_max=attack_config.clamp_max,
        )

    loss = None
    with torch.enable_grad():
        with freeze_model_parameters(model):
            for _ in range(attack_config.steps):
                model.zero_grad(set_to_none=True)
                adversarial = adversarial.detach()
                adversarial.requires_grad_(True)

                logits = model(adversarial)
                loss = F.cross_entropy(logits, labels)
                loss.backward()

                gradients = adversarial.grad
                if gradients is None:
                    raise RuntimeError(
                        "PGD attack expected input gradients, but none were produced."
                    )

                updated = adversarial + attack_config.resolved_alpha * gradients.sign()
                perturbation = (updated - clean_waveforms).clamp(
                    min=-attack_config.epsilon,
                    max=attack_config.epsilon,
                )
                adversarial = clean_waveforms + perturbation
                adversarial = _clamp_waveforms(
                    waveforms=adversarial,
                    clamp_min=attack_config.clamp_min,
                    clamp_max=attack_config.clamp_max,
                )

    adversarial = adversarial.detach()

    stats = {}
    if return_stats:
        if loss is None:
            raise RuntimeError("PGD attack did not execute any attack steps.")
        stats = _summarize_attack_stats(
            clean_waveforms=clean_waveforms,
            adversarial_waveforms=adversarial,
            loss=loss,
            extra_fields={
                "epsilon": float(attack_config.epsilon),
                "steps": float(attack_config.steps),
                "alpha": float(attack_config.resolved_alpha),
            },
        )

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
