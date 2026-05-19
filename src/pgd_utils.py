from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PGD_STEPS = 5
DEFAULT_PGD_RANDOM_START = False
DEFAULT_CLAMP_MIN = -1.0
DEFAULT_CLAMP_MAX = 1.0
PGD_ATTACK_NAME = "pgd"
PGD_ATTACK_TYPE = "untargeted_linf"


def format_attack_float_tag(value: float) -> str:
    value_text = format(value, "g")
    return value_text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class PgdAttackConfig:
    epsilon: float
    steps: int = DEFAULT_PGD_STEPS
    alpha: Optional[float] = None
    random_start: bool = DEFAULT_PGD_RANDOM_START
    clamp_min: float = DEFAULT_CLAMP_MIN
    clamp_max: float = DEFAULT_CLAMP_MAX
    save_adv_audio: bool = False

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}.")
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}.")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}.")
        if self.epsilon == 0 and self.alpha not in (None, 0, 0.0):
            raise ValueError("alpha must be 0 when epsilon is 0.")
        if self.epsilon > 0 and self.alpha == 0:
            raise ValueError("alpha must be positive when epsilon is greater than 0.")
        if self.clamp_min > self.clamp_max:
            raise ValueError(
                f"Invalid clamp range: clamp_min ({self.clamp_min}) > clamp_max ({self.clamp_max})."
            )

    @property
    def resolved_alpha(self) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        return float(self.epsilon) / float(self.steps)

    @property
    def artifact_stem(self) -> str:
        epsilon_tag = format_attack_float_tag(self.epsilon)
        return f"{PGD_ATTACK_NAME}_eps_{epsilon_tag}_steps_{self.steps}"

    def to_summary_fields(self) -> Dict[str, Any]:
        return {
            "attack_name": PGD_ATTACK_NAME,
            "attack_type": PGD_ATTACK_TYPE,
            "epsilon": float(self.epsilon),
            "steps": int(self.steps),
            "alpha": float(self.resolved_alpha),
            "random_start": bool(self.random_start),
            "clamp_min": float(self.clamp_min),
            "clamp_max": float(self.clamp_max),
            "save_adv_audio": bool(self.save_adv_audio),
        }


@dataclass(frozen=True)
class PgdArtifactContract:
    run_dir: Path
    clean_score_path: Path
    adv_score_path: Path
    defended_score_path: Path
    clean_metrics_path: Path
    adv_metrics_path: Path
    defended_metrics_path: Path
    summary_json_path: Path
    summary_text_path: Path
    adv_audio_dir: Optional[Path]


def build_pgd_artifact_contract(
    run_dir: Path,
    attack_config: PgdAttackConfig,
    clean_score_filename: str = "clean_scores.txt",
    adv_score_filename: Optional[str] = None,
) -> PgdArtifactContract:
    artifact_stem = attack_config.artifact_stem
    return PgdArtifactContract(
        run_dir=run_dir,
        clean_score_path=run_dir / clean_score_filename,
        adv_score_path=run_dir / (adv_score_filename or f"{artifact_stem}_scores.txt"),
        defended_score_path=run_dir / f"{artifact_stem}_defended_scores.txt",
        clean_metrics_path=run_dir / "clean_metrics.txt",
        adv_metrics_path=run_dir / f"{artifact_stem}_metrics.txt",
        defended_metrics_path=run_dir / f"{artifact_stem}_defended_metrics.txt",
        summary_json_path=run_dir / "pgd_metrics_summary.json",
        summary_text_path=run_dir / "pgd_metrics_summary.txt",
        adv_audio_dir=run_dir / f"{artifact_stem}_audio" if attack_config.save_adv_audio else None,
    )


def build_pgd_summary_stub(
    *,
    split: str,
    architecture: str,
    checkpoint_path: Path,
    dataset_root: Path,
    trial_file: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    attack_config: PgdAttackConfig,
    artifacts: PgdArtifactContract,
) -> Dict[str, Any]:
    summary = {
        "split": split,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "trial_file": str(trial_file),
        "output_dir": str(output_dir),
        "device": device,
        "batch_size": batch_size,
        "clean_score_path": str(artifacts.clean_score_path),
        "adv_score_path": str(artifacts.adv_score_path),
        "defended_score_path": str(artifacts.defended_score_path),
        "clean_metrics_path": str(artifacts.clean_metrics_path),
        "adv_metrics_path": str(artifacts.adv_metrics_path),
        "defended_metrics_path": str(artifacts.defended_metrics_path),
        "adv_audio_dir": str(artifacts.adv_audio_dir) if artifacts.adv_audio_dir else None,
    }
    summary.update(attack_config.to_summary_fields())
    return summary


def build_adv_metric_summary(adv_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        "min_dcf": {
            "adversarial": adv_metrics["min_dcf"],
        },
        "eer": {
            "adversarial": adv_metrics["eer"],
        },
        "cllr": {
            "adversarial": adv_metrics["cllr"],
        },
    }
