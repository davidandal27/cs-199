from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DRIVE_MOUNT_ROOT = Path("/content/drive")


@dataclass
class WorkflowPaths:
    dataset_root: Path
    metadata_root: Path
    train_audio_root: Optional[Path]
    dev_audio_root: Optional[Path]
    eval_audio_root: Optional[Path]
    train_metadata: Optional[Path]
    dev_metadata: Optional[Path]
    eval_metadata: Optional[Path]
    ssl_pretrained_path: Path
    output_dir: Path
    model_weights_path: Optional[Path] = None
    musan_path: Optional[Path] = None
    rir_path: Optional[Path] = None


def resolve_workflow_paths(
    config: Dict[str, Any],
    output_dir: str,
    model_weights_path: Optional[str] = None,
    require_training_assets: bool = True,
    require_dev_assets: bool = True,
    require_eval_assets: bool = True,
    train_audio_root_override: Optional[str] = None,
    dev_audio_root_override: Optional[str] = None,
    eval_audio_root_override: Optional[str] = None,
    train_metadata_override: Optional[str] = None,
    dev_metadata_override: Optional[str] = None,
    eval_metadata_override: Optional[str] = None,
) -> WorkflowPaths:
    dataset_root = _resolve_dir(config["database_path"], "dataset root")
    metadata_root = _resolve_dir(
        config.get("metadata_path", config["database_path"]),
        "metadata root",
    )

    paths = WorkflowPaths(
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        train_audio_root=(
            _resolve_dir(train_audio_root_override, "train audio root")
            if train_audio_root_override is not None
            else (
                _resolve_child_dir(dataset_root, "flac_T", "train audio root")
                if require_training_assets
                else None
            )
        ),
        dev_audio_root=(
            _resolve_dir(dev_audio_root_override, "dev audio root")
            if dev_audio_root_override is not None
            else (
                _resolve_child_dir(dataset_root, "flac_D", "dev audio root")
                if require_dev_assets
                else None
            )
        ),
        eval_audio_root=(
            _resolve_dir(eval_audio_root_override, "eval audio root")
            if eval_audio_root_override is not None
            else (
                _resolve_child_dir(
                    dataset_root,
                    "eval_full/flac_E_eval",
                    "eval audio root",
                )
                if require_eval_assets
                else None
            )
        ),
        train_metadata=(
            _resolve_file(train_metadata_override, "train metadata file")
            if train_metadata_override is not None
            else (
                _resolve_child_file(
                    metadata_root,
                    "ASVspoof5.train.tsv",
                    "train metadata file",
                )
                if require_training_assets
                else None
            )
        ),
        dev_metadata=(
            _resolve_file(dev_metadata_override, "dev metadata file")
            if dev_metadata_override is not None
            else (
                _resolve_child_file(
                    metadata_root,
                    "ASVspoof5.dev.track_1.tsv",
                    "dev metadata file",
                )
                if require_dev_assets
                else None
            )
        ),
        eval_metadata=(
            _resolve_file(eval_metadata_override, "eval metadata file")
            if eval_metadata_override is not None
            else (
                _resolve_child_file(
                    metadata_root,
                    "ASVspoof5.eval.track_1.tsv",
                    "eval metadata file",
                )
                if require_eval_assets
                else None
            )
        ),
        ssl_pretrained_path=_resolve_file(
            config["model_config"].get(
                "ssl_pretrained_path",
                "pretrained_models/WavLM-Large.pt",
            ),
            "WavLM checkpoint",
        ),
        output_dir=_prepare_output_dir(output_dir),
        model_weights_path=(
            _resolve_file(model_weights_path, "model weights")
            if model_weights_path not in (None, "", True)
            else None
        ),
    )

    if _str_to_bool(config.get("add_noise", False)) and require_training_assets:
        paths.musan_path = _resolve_dir(
            config.get("musan_path", "musan_data"),
            "MUSAN directory",
        )
        paths.rir_path = _resolve_dir(
            config.get("rir_path", "RIR_data"),
            "RIR directory",
        )

    return paths


def apply_path_overrides(config: Dict[str, Any], paths: WorkflowPaths) -> Dict[str, Any]:
    config["database_path"] = str(paths.dataset_root)
    config["metadata_path"] = str(paths.metadata_root)
    if paths.musan_path is not None:
        config["musan_path"] = str(paths.musan_path)
    if paths.rir_path is not None:
        config["rir_path"] = str(paths.rir_path)
    config["model_config"]["ssl_pretrained_path"] = str(paths.ssl_pretrained_path)
    return config


def _resolve_child_dir(root: Path, relative_path: str, label: str) -> Path:
    return _resolve_dir(root / relative_path, label)


def _resolve_child_file(root: Path, relative_path: str, label: str) -> Path:
    return _resolve_file(root / relative_path, label)


def _resolve_dir(path_value: Any, label: str) -> Path:
    path = _normalize_path(path_value, label)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected {label} directory at '{path}', but it was not found."
        )
    return path


def _resolve_file(path_value: Any, label: str) -> Path:
    path = _normalize_path(path_value, label)
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected {label} file at '{path}', but it was not found."
        )
    return path


def _prepare_output_dir(path_value: Any) -> Path:
    path = _normalize_path(path_value, "output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_path(path_value: Any, label: str) -> Path:
    if path_value in (None, ""):
        raise ValueError(f"Missing required path for {label}.")
    path = Path(path_value).expanduser()
    if _looks_like_drive_path(path):
        _ensure_drive_is_mounted(path)
    return path.resolve(strict=False)


def _looks_like_drive_path(path: Path) -> bool:
    return str(path).startswith(str(DRIVE_MOUNT_ROOT))


def _ensure_drive_is_mounted(path: Path) -> None:
    if not DRIVE_MOUNT_ROOT.exists():
        raise FileNotFoundError(
            "Google Drive is not mounted. Run "
            "drive.mount('/content/drive') before using "
            f"Drive-backed path '{path}'."
        )

    my_drive = DRIVE_MOUNT_ROOT / "MyDrive"
    if not my_drive.exists():
        raise FileNotFoundError(
            "Google Drive mount is incomplete. Expected '/content/drive/MyDrive' "
            f"before using '{path}'."
        )


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
