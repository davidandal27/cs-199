from pathlib import Path
from typing import Any, Tuple

import numpy as np
import soundfile as sf


def ensure_mono_audio(audio: np.ndarray, source: Any) -> np.ndarray:
    audio = np.asarray(audio)

    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        # soundfile returns multi-channel audio as (frames, channels)
        return audio.mean(axis=1)

    raise ValueError(
        f"Unsupported audio shape {audio.shape} from '{source}'. Expected 1D mono or 2D multi-channel audio."
    )


def read_audio_file(audio_path: Any, **read_kwargs: Any) -> Tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: '{path}'")

    try:
        audio, sample_rate = sf.read(str(path), **read_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file '{path}': {exc}") from exc

    return ensure_mono_audio(audio, path), sample_rate
