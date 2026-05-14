import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


DEFAULT_DEFENSE_CONFIG = {
    "sigma": 0.001,
    "normalize": True,
    "clamp": True,
    "samples": 1,
}


def get_default_defense_config() -> Dict[str, Any]:
    return dict(DEFAULT_DEFENSE_CONFIG)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def normalize_defense_config(defense_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = get_default_defense_config()
    if defense_config:
        normalized.update(defense_config)

    normalized["sigma"] = float(normalized["sigma"])
    normalized["normalize"] = _coerce_bool(normalized["normalize"])
    normalized["clamp"] = _coerce_bool(normalized["clamp"])
    normalized["samples"] = int(normalized["samples"])

    if normalized["sigma"] < 0:
        raise ValueError(
            f"Defense sigma must be non-negative, got {normalized['sigma']}."
        )
    if normalized["samples"] < 1:
        raise ValueError(
            f"Defense samples must be at least 1, got {normalized['samples']}."
        )
    return normalized


def resolve_defense_config_path(
    defense_config_path: str,
    config_path: Optional[str] = None,
) -> Path:
    path = Path(defense_config_path).expanduser()
    # Prefer the caller's explicit relative path when it already exists from the
    # current working directory, and only rebase onto the config directory as a fallback.
    if not path.is_absolute() and config_path is not None and not path.is_file():
        path = Path(config_path).expanduser().resolve(strict=False).parent / path
    return path.resolve(strict=False)


def load_defense_config_file(
    defense_config_path: str,
    config_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    resolved_path = resolve_defense_config_path(
        defense_config_path=defense_config_path,
        config_path=config_path,
    )
    with open(resolved_path, "r") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Defense config '{resolved_path}' must contain a JSON object."
        )

    if "defense" in payload and isinstance(payload["defense"], dict):
        payload = payload["defense"]

    return normalize_defense_config(payload), resolved_path


def apply_resolved_defense_config(
    config: Dict[str, Any],
    config_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_config = dict(config)
    selected_path = defense_config_path or resolved_config.get("defense_config_path")

    if selected_path:
        defense_config, resolved_path = load_defense_config_file(
            defense_config_path=selected_path,
            config_path=config_path,
        )
        resolved_config["defense_config_path"] = str(resolved_path)
    else:
        defense_config = normalize_defense_config(resolved_config.get("defense"))

    resolved_config["defense"] = defense_config
    return resolved_config


def get_defense_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    defense_config = normalize_defense_config(config.get("defense"))
    return {
        "sigma": defense_config["sigma"],
        "normalize": defense_config["normalize"],
        "clamp": defense_config["clamp"],
    }


def get_defense_samples(config: Dict[str, Any]) -> int:
    defense_config = normalize_defense_config(config.get("defense"))
    return defense_config["samples"]


def preprocess_audio(wav):
    wav = wav / (wav.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
    return wav



def randomized_smoothing(wav, sigma=0.001, clamp=True):
    if sigma <= 0:
        if clamp:
            return torch.clamp(wav, -1.0, 1.0)
        return wav

    noise = torch.randn_like(wav) * sigma
    wav = wav + noise
    if clamp:
        wav = torch.clamp(wav, -1.0, 1.0)
    return wav



def defend_audio(wav, sigma=0.001, normalize=True, clamp=True):
    if normalize:
        wav = preprocess_audio(wav)
    elif clamp:
        wav = torch.clamp(wav, -1.0, 1.0)

    wav = randomized_smoothing(wav, sigma=sigma, clamp=clamp)
    return wav


def _prepare_defense_input(wav, normalize=True, clamp=True):
    if normalize:
        return preprocess_audio(wav)
    if clamp:
        return torch.clamp(wav, -1.0, 1.0)
    return wav


def _defend_audio_samples_vectorized(
    wav,
    defense_samples: int,
    sigma=0.001,
    normalize=True,
    clamp=True,
):
    prepared_wav = _prepare_defense_input(
        wav,
        normalize=normalize,
        clamp=clamp,
    )
    defended_wavs = prepared_wav.unsqueeze(0).repeat(defense_samples, 1, 1)

    if sigma > 0:
        defended_wavs = defended_wavs + torch.randn_like(defended_wavs) * sigma

    if clamp:
        defended_wavs = torch.clamp(defended_wavs, -1.0, 1.0)

    return defended_wavs


def forward_with_defense(
    model,
    wav,
    defense_kwargs: Optional[Dict[str, Any]] = None,
    defense_samples: int = 1,
    vectorized: bool = False,
):
    defense_kwargs = defense_kwargs or {}
    if defense_samples < 1:
        raise ValueError(
            f"defense_samples must be at least 1, got {defense_samples}."
        )

    if vectorized and defense_samples > 1:
        defended_wavs = _defend_audio_samples_vectorized(
            wav,
            defense_samples=defense_samples,
            **defense_kwargs,
        )
        batch_size = wav.shape[0]
        accumulated_logits = model(defended_wavs.reshape(defense_samples * batch_size, -1))
        averaged_logits = accumulated_logits.reshape(defense_samples, batch_size, -1).mean(dim=0)
        return averaged_logits, defended_wavs[0]

    defended_wav = defend_audio(wav, **defense_kwargs)
    accumulated_logits = model(defended_wav)

    for _ in range(defense_samples - 1):
        accumulated_logits += model(defend_audio(wav, **defense_kwargs))

    return accumulated_logits.div(defense_samples), defended_wav
