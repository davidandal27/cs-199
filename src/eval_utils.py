import json
import math
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.path_utils import apply_path_overrides, resolve_workflow_paths
from src.defense_utils import (
    apply_resolved_defense_config,
    forward_with_defense,
    get_defense_kwargs,
    get_defense_samples,
)


def resolve_eval_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn(
            "GPU not detected; falling back to CPU. Evaluation may be slow.",
            RuntimeWarning,
            stacklevel=2,
        )
    return device


def load_eval_config(
    config_path: str,
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    ssl_pretrained_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        config = json.load(file)

    if dataset_root is not None:
        config["database_path"] = dataset_root
    if metadata_root is not None:
        config["metadata_path"] = metadata_root
    if ssl_pretrained_path is not None:
        config["model_config"]["ssl_pretrained_path"] = ssl_pretrained_path

    return apply_resolved_defense_config(
        config=config,
        config_path=config_path,
        defense_config_path=defense_config_path,
    )


def apply_eval_path_fallbacks(
    config: Dict[str, Any],
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    trial_file: Optional[str] = None,
    audio_root: Optional[str] = None,
) -> Dict[str, Any]:
    if dataset_root is None and audio_root is not None:
        config["database_path"] = str(Path(audio_root).expanduser().resolve(strict=False).parent)
    if metadata_root is None and trial_file is not None:
        config["metadata_path"] = str(Path(trial_file).expanduser().resolve(strict=False).parent)
    return config


def build_eval_loader(
    paths,
    batch_size: int,
    split: str,
    trial_path: Optional[Path] = None,
):
    import torch
    from torch.utils.data import DataLoader
    from src.data_utils import TestDataset, genSpoof_list

    default_trial_path, audio_root = resolve_split_paths(paths=paths, split=split)
    trial_path = trial_path or default_trial_path

    _, utterance_ids = genSpoof_list(dir_meta=trial_path)
    validate_trial_audio_files(
        utterance_ids=utterance_ids,
        audio_root=audio_root,
        trial_path=trial_path,
    )
    eval_set = TestDataset(list_IDs=utterance_ids, base_dir=audio_root)
    data_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return data_loader, trial_path


def build_labeled_eval_loader(
    paths,
    batch_size: int,
    split: str,
    trial_path: Optional[Path] = None,
    return_trial_line: bool = False,
):
    from torch.utils.data import DataLoader
    import torch
    from src.data_utils import LabeledEvalDataset, load_trial_records

    default_trial_path, audio_root = resolve_split_paths(paths=paths, split=split)
    trial_path = trial_path or default_trial_path

    trial_records = load_trial_records(trial_path)
    validate_trial_audio_files(
        utterance_ids=[record["utterance_id"] for record in trial_records],
        audio_root=audio_root,
        trial_path=trial_path,
    )
    eval_set = LabeledEvalDataset(
        trial_records=trial_records,
        base_dir=audio_root,
        return_trial_line=return_trial_line,
    )
    data_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return data_loader, trial_path, trial_records


def resolve_split_paths(paths, split: str):
    if split == "dev":
        if paths.dev_metadata is None or paths.dev_audio_root is None:
            raise ValueError("Dev split assets are not configured.")
        return paths.dev_metadata, paths.dev_audio_root
    if split == "eval":
        if paths.eval_metadata is None or paths.eval_audio_root is None:
            raise ValueError("Eval split assets are not configured.")
        return paths.eval_metadata, paths.eval_audio_root
    raise ValueError(f"Unsupported split '{split}'. Use 'dev' or 'eval'.")


def validate_trial_audio_files(
    utterance_ids: List[str],
    audio_root: Path,
    trial_path: Path,
) -> None:
    missing = []
    for utt_id in utterance_ids:
        if not (audio_root / f"{utt_id}.flac").is_file():
            missing.append(utt_id)
            if len(missing) >= 10:
                break

    if not missing:
        return

    raise FileNotFoundError(
        "Trial file and audio directory do not match. "
        f"Missing {len(missing)} or more files from '{audio_root}' referenced by "
        f"'{trial_path}'. Example utterance IDs: {', '.join(missing)}. "
        "Check that --trial-file and --audio-root point to the same dataset split "
        "and that the audio directory is complete."
    )


def build_model(model_config: Dict[str, Any], device):
    module = import_module(f"models.{model_config['architecture']}")
    model_class = getattr(module, "Model")
    model = model_class(model_config, device=device).to(device)
    return model


def load_plain_model_weights(model, weights_path, device) -> None:
    import torch

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)


def canonicalize_scores_by_trial_records(
    trial_records,
    utterance_ids: Sequence[str],
    scores: Sequence[float],
    *,
    allow_duplicate_utterances: bool = False,
) -> Tuple[List[str], List[float]]:
    if len(utterance_ids) != len(scores):
        raise ValueError(
            "Utterance and score counts must match before score ordering."
        )

    expected_utterance_ids = [record["utterance_id"] for record in trial_records]
    if allow_duplicate_utterances and len(expected_utterance_ids) != len(set(expected_utterance_ids)):
        raise ValueError(
            "Distributed score merging requires unique utterance IDs in the trial file."
        )

    expected_utterance_id_set = set(expected_utterance_ids)
    extra_utterance_ids = sorted(set(utterance_ids) - expected_utterance_id_set)
    if extra_utterance_ids:
        preview = ", ".join(extra_utterance_ids[:5])
        raise ValueError(
            "Scoring produced utterance IDs not present in the trial file: "
            f"{preview}."
        )

    score_by_utterance_id: Dict[str, float] = {}
    for utterance_id, score in zip(utterance_ids, scores):
        normalized_score = float(score)
        if utterance_id in score_by_utterance_id:
            if not allow_duplicate_utterances:
                raise ValueError(
                    "Duplicate utterance IDs were produced during score collection: "
                    f"{utterance_id}."
                )
            if not math.isclose(
                score_by_utterance_id[utterance_id],
                normalized_score,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "Duplicate utterance IDs were gathered with conflicting scores: "
                    f"{utterance_id}."
                )
            continue
        score_by_utterance_id[utterance_id] = normalized_score

    missing_utterance_ids = [
        utterance_id
        for utterance_id in expected_utterance_ids
        if utterance_id not in score_by_utterance_id
    ]
    if missing_utterance_ids:
        preview = ", ".join(missing_utterance_ids[:5])
        raise ValueError(
            "Scoring did not produce scores for every trial utterance. "
            f"Missing examples: {preview}."
        )

    ordered_scores = [
        score_by_utterance_id[utterance_id]
        for utterance_id in expected_utterance_ids
    ]
    return expected_utterance_ids, ordered_scores


def merge_gathered_score_payloads(
    trial_records,
    gathered_payloads: Sequence[Dict[str, Sequence[float]]],
) -> Tuple[List[str], List[float]]:
    merged_utterance_ids: List[str] = []
    merged_scores: List[float] = []

    for payload in gathered_payloads:
        if payload is None:
            continue
        payload_utterance_ids = payload.get("utterance_ids")
        payload_scores = payload.get("scores")
        if payload_utterance_ids is None or payload_scores is None:
            raise ValueError(
                "Gathered score payloads must contain 'utterance_ids' and 'scores'."
            )
        if len(payload_utterance_ids) != len(payload_scores):
            raise ValueError(
                "Gathered score payload utterance and score counts must match."
            )
        merged_utterance_ids.extend(payload_utterance_ids)
        merged_scores.extend(payload_scores)

    return canonicalize_scores_by_trial_records(
        trial_records=trial_records,
        utterance_ids=merged_utterance_ids,
        scores=merged_scores,
        allow_duplicate_utterances=True,
    )


def generate_adversarial_batch(
    model,
    batch_x,
    batch_y,
    epsilon: float,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    return_stats: bool = True,
):
    from src.attack_utils import generate_fgsm_batch

    return generate_fgsm_batch(
        model=model,
        waveforms=batch_x,
        labels=batch_y,
        epsilon=epsilon,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        return_stats=return_stats,
    )


def format_epsilon_tag(epsilon: float) -> str:
    epsilon_text = format(epsilon, "g")
    return epsilon_text.replace("-", "m").replace(".", "p")


def _collect_scores(
    data_loader,
    model,
    device,
    attack: bool = False,
    epsilon: float = 0.0,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    defense_kwargs: Optional[Dict[str, Any]] = None,
    defense_samples: int = 1,
    adversarial_audio_dir: Optional[Path] = None,
):
    import torch
    from tqdm import tqdm

    from src.attack_utils import save_adversarial_waveforms

    model.eval()
    utterance_ids: List[str] = []
    scores: List[float] = []
    attack_stats: List[Dict[str, float]] = []
    defense_kwargs = defense_kwargs or {}

    for batch_x, batch_y, batch_utt_ids in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        if attack:
            batch_x, batch_stats = generate_adversarial_batch(
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                epsilon=epsilon,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                return_stats=True,
            )
            attack_stats.append(batch_stats)

        with torch.no_grad():
            batch_out, _ = forward_with_defense(
                model=model,
                wav=batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
            )
            batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()

        if attack and adversarial_audio_dir is not None:
            save_adversarial_waveforms(
                waveforms=batch_x,
                utterance_ids=batch_utt_ids,
                output_dir=adversarial_audio_dir,
            )

        utterance_ids.extend(batch_utt_ids)
        scores.extend(batch_scores.tolist())

    return utterance_ids, scores, attack_stats


def _average_attack_stats(attack_stats: List[Dict[str, float]]) -> Dict[str, float]:
    if not attack_stats:
        return {}

    summary = {}
    keys = attack_stats[0].keys()
    for key in keys:
        summary[key] = sum(batch_stats[key] for batch_stats in attack_stats) / len(attack_stats)
    return summary


def _write_ordered_score_lines(
    save_path: Path,
    trial_records,
    utterance_ids: List[str],
    scores: List[float],
) -> None:
    utterance_ids, scores = canonicalize_scores_by_trial_records(
        trial_records=trial_records,
        utterance_ids=utterance_ids,
        scores=scores,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as score_file:
        for record, utt_id, score in zip(trial_records, utterance_ids, scores):
            score_file.write(
                f"{record['speaker_id']} {record['utterance_id']} {score} {record['label_name']}\n"
            )


def compute_cm_metrics_from_trial_records(
    trial_records,
    utterance_ids: List[str],
    scores: List[float],
) -> Dict[str, float]:
    import numpy as np

    from eval.calculate_modules import calculate_CLLR, compute_eer, compute_mindcf

    utterance_ids, scores = canonicalize_scores_by_trial_records(
        trial_records=trial_records,
        utterance_ids=utterance_ids,
        scores=scores,
    )
    ordered_labels = []
    ordered_scores = []
    for record, utt_id, score in zip(trial_records, utterance_ids, scores):
        ordered_labels.append(record["label_name"])
        ordered_scores.append(score)

    cm_scores = np.asarray(ordered_scores, dtype=np.float64)
    cm_labels = np.asarray(ordered_labels, dtype=str)
    bona_cm = cm_scores[cm_labels == "bonafide"]
    spoof_cm = cm_scores[cm_labels == "spoof"]

    if bona_cm.size == 0 or spoof_cm.size == 0:
        raise ValueError(
            "Metric computation requires both bonafide and spoof trials in the score set."
        )

    p_spoof = 0.05
    eer, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)
    cllr = calculate_CLLR(bona_cm, spoof_cm)
    min_dcf, _ = compute_mindcf(
        frr=frr,
        far=far,
        thresholds=thresholds,
        Pspoof=p_spoof,
        Cmiss=1,
        Cfa=10,
    )
    return {"min_dcf": min_dcf, "eer": eer, "cllr": cllr}


def _try_write_artifact(
    action,
    description: str,
    artifact_warnings: List[str],
) -> bool:
    try:
        action()
        return True
    except OSError as exc:
        message = f"Unable to write {description}: {exc}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        artifact_warnings.append(message)
        return False


def write_score_file(
    data_loader,
    model,
    device,
    save_path: Path,
    trial_path: Path,
) -> None:
    import torch
    from tqdm import tqdm

    model.eval()
    trial_records = load_trial_records(trial_path)

    utterance_ids = []
    scores = []
    for batch_x, batch_utt_ids in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
            batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()
        utterance_ids.extend(batch_utt_ids)
        scores.extend(batch_scores.tolist())

    _write_ordered_score_lines(
        save_path=save_path,
        trial_records=trial_records,
        utterance_ids=utterance_ids,
        scores=scores,
    )


def write_metric_report(
    metrics_path: Path,
    min_dcf: float,
    eer: float,
    cllr: float,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as report:
        report.write("CM SYSTEM\n")
        report.write(
            f"min DCF = {min_dcf}\n"
            f"EER = {eer * 100:.9f} %\n"
            f"CLLR = {cllr * 100:.9f} %\n"
        )


def compute_cm_metrics(score_path: Path, metrics_path: Path) -> Dict[str, float]:
    from eval.calculate_metrics import calculate_minDCF_EER_CLLR

    min_dcf, eer, cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=score_path,
        output_file=metrics_path,
        printout=False,
    )
    write_metric_report(metrics_path, min_dcf, eer, cllr)
    return {"min_dcf": min_dcf, "eer": eer, "cllr": cllr}


def read_metric_report(metrics_path: Path) -> Dict[str, float]:
    metrics = {}
    with open(metrics_path, "r") as report:
        for line in report:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "min DCF":
                metrics["min_dcf"] = float(value)
            elif key == "EER":
                metrics["eer"] = float(value.split(" ")[0]) / 100.0
            elif key == "CLLR":
                metrics["cllr"] = float(value.split(" ")[0]) / 100.0
    return metrics


def _relative_delta(clean_value: float, adv_value: float) -> Optional[float]:
    if clean_value == 0:
        return None
    return (adv_value - clean_value) / clean_value


def build_metric_delta_summary(
    clean_metrics: Dict[str, float],
    adv_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        "min_dcf": {
            "clean": clean_metrics["min_dcf"],
            "adversarial": adv_metrics["min_dcf"],
            "absolute_delta": adv_metrics["min_dcf"] - clean_metrics["min_dcf"],
            "relative_delta": _relative_delta(
                clean_metrics["min_dcf"], adv_metrics["min_dcf"]
            ),
        },
        "eer": {
            "clean": clean_metrics["eer"],
            "adversarial": adv_metrics["eer"],
            "absolute_delta": adv_metrics["eer"] - clean_metrics["eer"],
            "relative_delta": _relative_delta(clean_metrics["eer"], adv_metrics["eer"]),
        },
        "cllr": {
            "clean": clean_metrics["cllr"],
            "adversarial": adv_metrics["cllr"],
            "absolute_delta": adv_metrics["cllr"] - clean_metrics["cllr"],
            "relative_delta": _relative_delta(
                clean_metrics["cllr"], adv_metrics["cllr"]
            ),
        },
    }


def write_metric_summary_json(summary_path: Path, summary: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")


def write_metric_summary_text(summary_path: Path, summary: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = {
        "min_dcf": "minDCF",
        "eer": "EER",
        "cllr": "CLLR",
    }
    with open(summary_path, "w") as summary_file:
        summary_file.write("FGSM METRIC SUMMARY\n")
        summary_file.write(f"split = {summary['split']}\n")
        summary_file.write(f"architecture = {summary['architecture']}\n")
        summary_file.write(f"checkpoint_path = {summary['checkpoint_path']}\n")
        summary_file.write(f"dataset_root = {summary['dataset_root']}\n")
        summary_file.write(f"trial_file = {summary['trial_file']}\n")
        summary_file.write(f"output_dir = {summary['output_dir']}\n")
        summary_file.write(f"device = {summary['device']}\n")
        summary_file.write(f"batch_size = {summary['batch_size']}\n")
        summary_file.write(f"defense_samples = {summary['defense_samples']}\n")
        summary_file.write(f"epsilon = {summary['epsilon']}\n")
        summary_file.write(f"clamp_min = {summary['clamp_min']}\n")
        summary_file.write(f"clamp_max = {summary['clamp_max']}\n")
        summary_file.write(f"clean_score_file = {summary['clean_score_path']}\n")
        summary_file.write(f"adversarial_score_file = {summary['adv_score_path']}\n")
        summary_file.write(f"clean_metric_file = {summary['clean_metrics_path']}\n")
        summary_file.write(f"adversarial_metric_file = {summary['adv_metrics_path']}\n")
        summary_file.write(f"adversarial_audio_dir = {summary['adv_audio_dir']}\n")
        summary_file.write(f"utterance_count = {summary['utterance_count']}\n")
        summary_file.write("\n")
        for metric_key in ("min_dcf", "eer", "cllr"):
            metric = summary["metrics"][metric_key]
            label = metric_names[metric_key]
            clean_value = metric["clean"]
            adv_value = metric["adversarial"]
            abs_delta = metric["absolute_delta"]
            rel_delta = metric["relative_delta"]
            if metric_key in ("eer", "cllr"):
                clean_text = f"{clean_value * 100:.9f}%"
                adv_text = f"{adv_value * 100:.9f}%"
                abs_text = f"{abs_delta * 100:.9f}%"
            else:
                clean_text = f"{clean_value:.9f}"
                adv_text = f"{adv_value:.9f}"
                abs_text = f"{abs_delta:.9f}"
            rel_text = "n/a" if rel_delta is None else f"{rel_delta * 100:.9f}%"
            summary_file.write(
                f"{label}: clean={clean_text}, adversarial={adv_text}, "
                f"absolute_delta={abs_text}, relative_delta={rel_text}\n"
            )


def run_clean_evaluation(
    config_path: str,
    weights_path: str,
    output_dir: str,
    split: str = "eval",
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    trial_file: Optional[str] = None,
    audio_root: Optional[str] = None,
    ssl_pretrained_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    score_filename: Optional[str] = None,
    metrics_only: bool = False,
    seed: int = 1234,
) -> Dict[str, Any]:
    config = load_eval_config(
        config_path=config_path,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        ssl_pretrained_path=ssl_pretrained_path,
        defense_config_path=defense_config_path,
    )
    config = apply_eval_path_fallbacks(
        config=config,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        trial_file=trial_file,
        audio_root=audio_root,
    )
    import torch

    paths = resolve_workflow_paths(
        config=config,
        output_dir=output_dir,
        model_weights_path=weights_path,
        require_training_assets=False,
        require_dev_assets=split == "dev" and trial_file is None,
        require_eval_assets=split == "eval" and trial_file is None,
        dev_audio_root_override=audio_root if split == "dev" else None,
        eval_audio_root_override=audio_root if split == "eval" else None,
        dev_metadata_override=trial_file if split == "dev" else None,
        eval_metadata_override=trial_file if split == "eval" else None,
    )
    config = apply_path_overrides(config, paths)
    from src.utils import set_seed

    device = resolve_eval_device()
    set_seed(seed, config)
    defense_kwargs = get_defense_kwargs(config)
    defense_samples = get_defense_samples(config)

    model = build_model(config["model_config"], device)
    load_plain_model_weights(model, paths.model_weights_path, device)

    effective_batch_size = batch_size or config["batch_size"]
    eval_loader, trial_path, trial_records = build_labeled_eval_loader(
        paths=paths,
        batch_size=effective_batch_size,
        split=split,
    )

    run_dir = paths.output_dir / config["model_tag"] / f"{split}_clean_eval"
    score_path = run_dir / (score_filename or f"{split}_scores.txt")
    metrics_path = run_dir / f"{split}_metrics.txt"
    artifact_warnings: List[str] = []

    utterance_ids, scores, _ = _collect_scores(
        data_loader=eval_loader,
        model=model,
        device=device,
        defense_kwargs=defense_kwargs,
        defense_samples=defense_samples,
    )
    metrics = compute_cm_metrics_from_trial_records(
        trial_records=trial_records,
        utterance_ids=utterance_ids,
        scores=scores,
    )

    persisted_score_path = None
    persisted_metrics_path = None
    if not metrics_only:
        if _try_write_artifact(
            action=lambda: _write_ordered_score_lines(
                save_path=score_path,
                trial_records=trial_records,
                utterance_ids=utterance_ids,
                scores=scores,
            ),
            description=f"clean score file '{score_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_score_path = score_path
        if _try_write_artifact(
            action=lambda: write_metric_report(
                metrics_path=metrics_path,
                min_dcf=metrics["min_dcf"],
                eer=metrics["eer"],
                cllr=metrics["cllr"],
            ),
            description=f"clean metric report '{metrics_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_metrics_path = metrics_path

    return {
        "split": split,
        "device": str(device),
        "score_path": persisted_score_path,
        "metrics_path": persisted_metrics_path,
        "weights_path": paths.model_weights_path,
        "trial_path": trial_path,
        "batch_size": effective_batch_size,
        "defense_samples": defense_samples,
        "min_dcf": metrics["min_dcf"],
        "eer": metrics["eer"],
        "cllr": metrics["cllr"],
        "artifact_warnings": artifact_warnings,
    }


def run_fgsm_scoring_pipeline(
    config_path: str,
    weights_path: str,
    output_dir: str,
    split: str = "dev",
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    trial_file: Optional[str] = None,
    audio_root: Optional[str] = None,
    ssl_pretrained_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    epsilon: float = 0.001,
    clamp_min: Optional[float] = -1.0,
    clamp_max: Optional[float] = 1.0,
    clean_score_filename: str = "clean_scores.txt",
    adv_score_filename: Optional[str] = None,
    save_adv_audio: bool = False,
    metrics_only: bool = False,
    seed: int = 1234,
) -> Dict[str, Any]:
    config = load_eval_config(
        config_path=config_path,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        ssl_pretrained_path=ssl_pretrained_path,
        defense_config_path=defense_config_path,
    )
    config = apply_eval_path_fallbacks(
        config=config,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        trial_file=trial_file,
        audio_root=audio_root,
    )
    import torch

    paths = resolve_workflow_paths(
        config=config,
        output_dir=output_dir,
        model_weights_path=weights_path,
        require_training_assets=False,
        require_dev_assets=split == "dev" and trial_file is None,
        require_eval_assets=split == "eval" and trial_file is None,
        dev_audio_root_override=audio_root if split == "dev" else None,
        eval_audio_root_override=audio_root if split == "eval" else None,
        dev_metadata_override=trial_file if split == "dev" else None,
        eval_metadata_override=trial_file if split == "eval" else None,
    )
    config = apply_path_overrides(config, paths)
    from src.utils import set_seed

    device = resolve_eval_device()
    set_seed(seed, config)
    defense_kwargs = get_defense_kwargs(config)
    defense_samples = get_defense_samples(config)

    model = build_model(config["model_config"], device)
    load_plain_model_weights(model, paths.model_weights_path, device)

    effective_batch_size = batch_size or config["batch_size"]
    eval_loader, trial_path, trial_records = build_labeled_eval_loader(
        paths,
        effective_batch_size,
        split,
    )

    run_dir = paths.output_dir / config["model_tag"] / f"{split}_fgsm_eval"
    clean_score_path = run_dir / clean_score_filename
    clean_metrics_path = run_dir / "clean_metrics.txt"
    epsilon_tag = format_epsilon_tag(epsilon)
    adv_score_path = run_dir / (
        adv_score_filename or f"fgsm_eps_{epsilon_tag}_scores.txt"
    )
    adv_metrics_path = run_dir / f"fgsm_eps_{epsilon_tag}_metrics.txt"
    summary_json_path = run_dir / "fgsm_metrics_summary.json"
    summary_text_path = run_dir / "fgsm_metrics_summary.txt"
    adv_audio_dir = run_dir / f"fgsm_eps_{epsilon_tag}_audio" if save_adv_audio else None
    artifact_warnings: List[str] = []

    if metrics_only and save_adv_audio:
        message = (
            "Ignoring save_adv_audio because metrics_only=True disables run artifacts."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        artifact_warnings.append(message)
        adv_audio_dir = None

    clean_utterance_ids, clean_scores, _ = _collect_scores(
        data_loader=eval_loader,
        model=model,
        device=device,
        defense_kwargs=defense_kwargs,
        defense_samples=defense_samples,
    )
    clean_metrics = compute_cm_metrics_from_trial_records(
        trial_records=trial_records,
        utterance_ids=clean_utterance_ids,
        scores=clean_scores,
    )

    adv_utterance_ids, adv_scores, attack_stats = _collect_scores(
        data_loader=eval_loader,
        model=model,
        device=device,
        attack=True,
        epsilon=epsilon,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        defense_kwargs=defense_kwargs,
        defense_samples=defense_samples,
        adversarial_audio_dir=adv_audio_dir,
    )

    if clean_utterance_ids != adv_utterance_ids:
        raise ValueError("Clean and adversarial scoring produced different utterance orders.")

    adv_metrics = compute_cm_metrics_from_trial_records(
        trial_records=trial_records,
        utterance_ids=adv_utterance_ids,
        scores=adv_scores,
    )
    metric_summary = build_metric_delta_summary(clean_metrics, adv_metrics)
    summary = {
        "split": split,
        "architecture": config["model_config"]["architecture"],
        "checkpoint_path": str(paths.model_weights_path),
        "dataset_root": str(paths.dataset_root),
        "trial_file": str(trial_path),
        "output_dir": str(run_dir),
        "device": str(device),
        "batch_size": effective_batch_size,
        "defense_samples": defense_samples,
        "epsilon": epsilon,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "clean_score_path": str(clean_score_path),
        "adv_score_path": str(adv_score_path),
        "clean_metrics_path": str(clean_metrics_path),
        "adv_metrics_path": str(adv_metrics_path),
        "adv_audio_dir": str(adv_audio_dir) if adv_audio_dir is not None else None,
        "utterance_count": len(clean_utterance_ids),
        "metrics": metric_summary,
        "attack_stats": _average_attack_stats(attack_stats),
    }

    persisted_clean_score_path = None
    persisted_adv_score_path = None
    persisted_clean_metrics_path = None
    persisted_adv_metrics_path = None
    persisted_summary_json_path = None
    persisted_summary_text_path = None
    persisted_adv_audio_dir = adv_audio_dir

    if not metrics_only:
        if _try_write_artifact(
            action=lambda: _write_ordered_score_lines(
                save_path=clean_score_path,
                trial_records=trial_records,
                utterance_ids=clean_utterance_ids,
                scores=clean_scores,
            ),
            description=f"clean score file '{clean_score_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_clean_score_path = clean_score_path

        if _try_write_artifact(
            action=lambda: _write_ordered_score_lines(
                save_path=adv_score_path,
                trial_records=trial_records,
                utterance_ids=adv_utterance_ids,
                scores=adv_scores,
            ),
            description=f"adversarial score file '{adv_score_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_adv_score_path = adv_score_path

        if _try_write_artifact(
            action=lambda: write_metric_report(
                metrics_path=clean_metrics_path,
                min_dcf=clean_metrics["min_dcf"],
                eer=clean_metrics["eer"],
                cllr=clean_metrics["cllr"],
            ),
            description=f"clean metric report '{clean_metrics_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_clean_metrics_path = clean_metrics_path

        if _try_write_artifact(
            action=lambda: write_metric_report(
                metrics_path=adv_metrics_path,
                min_dcf=adv_metrics["min_dcf"],
                eer=adv_metrics["eer"],
                cllr=adv_metrics["cllr"],
            ),
            description=f"adversarial metric report '{adv_metrics_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_adv_metrics_path = adv_metrics_path

        if _try_write_artifact(
            action=lambda: write_metric_summary_json(summary_json_path, summary),
            description=f"FGSM summary JSON '{summary_json_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_summary_json_path = summary_json_path

        if _try_write_artifact(
            action=lambda: write_metric_summary_text(summary_text_path, summary),
            description=f"FGSM summary text '{summary_text_path}'",
            artifact_warnings=artifact_warnings,
        ):
            persisted_summary_text_path = summary_text_path
    else:
        persisted_adv_audio_dir = None

    return {
        "split": split,
        "device": str(device),
        "weights_path": paths.model_weights_path,
        "trial_path": trial_path,
        "batch_size": effective_batch_size,
        "defense_samples": defense_samples,
        "epsilon": epsilon,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "clean_score_path": persisted_clean_score_path,
        "adv_score_path": persisted_adv_score_path,
        "clean_metrics_path": persisted_clean_metrics_path,
        "adv_metrics_path": persisted_adv_metrics_path,
        "adv_audio_dir": persisted_adv_audio_dir,
        "summary_json_path": persisted_summary_json_path,
        "summary_text_path": persisted_summary_text_path,
        "utterance_count": len(clean_utterance_ids),
        "attack_stats": summary["attack_stats"],
        "metrics": metric_summary,
        "artifact_warnings": artifact_warnings,
    }
