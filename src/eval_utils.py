import json
import math
import shutil
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.distributed_eval import (
    DeterministicShardSampler,
    broadcast_from_primary,
    cleanup_distributed_eval_runtime,
    distributed_barrier,
    distributed_launch_requested,
    initialize_distributed_eval_runtime,
    is_primary_rank,
    rank_temp_dir,
    rank_temp_path,
    uses_multi_rank_distribution,
)
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
    distributed_runtime=None,
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
    sampler = None
    if uses_multi_rank_distribution(distributed_runtime):
        sampler = DeterministicShardSampler(
            eval_set,
            num_replicas=distributed_runtime.world_size,
            rank=distributed_runtime.rank,
        )
    data_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
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
    distributed_runtime=None,
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
    sampler = None
    if uses_multi_rank_distribution(distributed_runtime):
        sampler = DeterministicShardSampler(
            eval_set,
            num_replicas=distributed_runtime.world_size,
            rank=distributed_runtime.rank,
        )
    data_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
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

    score_by_utterance_id = {uid: float(sc) for uid, sc in zip(utterance_ids, scores)}
    missing_utterance_ids = set(expected_utterance_ids) - set(score_by_utterance_id.keys())
    if missing_utterance_ids:
        preview = ", ".join(list(missing_utterance_ids)[:5])
        raise ValueError(
            "Scoring did not produce scores for every trial utterance. "
            f"Missing examples: {preview}."
        )
    ordered_scores = [score_by_utterance_id[uid] for uid in expected_utterance_ids]
    return expected_utterance_ids, ordered_scores

def merge_gathered_score_payloads(
    trial_records,
    gathered_payloads: Sequence[Dict[str, Sequence[float]]],
    *,
    score_key: str = "scores",
) -> Tuple[List[str], List[float]]:
    merged_utterance_ids: List[str] = []
    merged_scores: List[float] = []

    for payload in gathered_payloads:
        if payload is None:
            continue
        payload_utterance_ids = payload.get("utterance_ids")
        payload_scores = payload.get(score_key)
        if payload_utterance_ids is None or payload_scores is None:
            raise ValueError(
                "Gathered score payloads must contain 'utterance_ids' and the "
                f"'{score_key}' score field."
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


def generate_pgd_adversarial_batch(
    model,
    batch_x,
    batch_y,
    epsilon: float,
    steps: int,
    alpha: Optional[float],
    random_start: bool,
    clamp_min: Optional[float] = -1.0,
    clamp_max: Optional[float] = 1.0,
    return_stats: bool = True,
):
    from src.attack_utils import generate_pgd_batch

    return generate_pgd_batch(
        model=model,
        waveforms=batch_x,
        labels=batch_y,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        random_start=random_start,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        return_stats=return_stats,
    )


def format_epsilon_tag(epsilon: float) -> str:
    from src.pgd_utils import format_attack_float_tag

    return format_attack_float_tag(epsilon)


def _collect_scores(
    data_loader,
    model,
    device,
    attack: bool = False,
    attack_name: str = "fgsm",
    epsilon: float = 0.0,
    steps: int = 5,
    alpha: Optional[float] = None,
    random_start: bool = False,
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
    use_non_blocking = device.type == "cuda"

    for batch_x, batch_y, batch_utt_ids in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=use_non_blocking)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        if attack:
            if attack_name == "fgsm":
                batch_x, batch_stats = generate_adversarial_batch(
                    model=model,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    epsilon=epsilon,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                    return_stats=True,
                )
            elif attack_name == "pgd":
                batch_x, batch_stats = generate_pgd_adversarial_batch(
                    model=model,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    epsilon=epsilon,
                    steps=steps,
                    alpha=alpha,
                    random_start=random_start,
                    clamp_min=-1.0 if clamp_min is None else clamp_min,
                    clamp_max=1.0 if clamp_max is None else clamp_max,
                    return_stats=True,
                )
            else:
                raise ValueError(f"Unsupported attack_name '{attack_name}'.")
            attack_stats.append(batch_stats)

        with torch.no_grad():
            batch_out, _ = forward_with_defense(
                model=model,
                wav=batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
                vectorized=defense_samples > 1,
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


def _collect_pgd_attack_scores(
    data_loader,
    model,
    device,
    epsilon: float,
    steps: int,
    alpha: Optional[float],
    random_start: bool,
    clamp_min: float,
    clamp_max: float,
    defense_kwargs: Optional[Dict[str, Any]] = None,
    defense_samples: int = 1,
    adversarial_audio_dir: Optional[Path] = None,
):
    import torch
    from tqdm import tqdm

    from src.attack_utils import save_adversarial_waveforms

    model.eval()
    utterance_ids: List[str] = []
    defended_scores: List[float] = []
    attack_stats: List[Dict[str, float]] = []
    defense_kwargs = defense_kwargs or {}
    use_non_blocking = device.type == "cuda"

    for batch_x, batch_y, batch_utt_ids in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=use_non_blocking)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # PGD is still generated against the plain model, but the emitted
        # attacked scores come only from the defended inference path.
        adv_batch_x, batch_stats = generate_pgd_adversarial_batch(
            model=model,
            batch_x=batch_x,
            batch_y=batch_y,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            random_start=random_start,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            return_stats=True,
        )
        attack_stats.append(batch_stats)

        with torch.no_grad():
            defended_batch_out, _ = forward_with_defense(
                model=model,
                wav=adv_batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
                vectorized=defense_samples > 1,
            )
            defended_batch_scores = defended_batch_out[:, 1].data.cpu().numpy().ravel()

        if adversarial_audio_dir is not None:
            save_adversarial_waveforms(
                waveforms=adv_batch_x,
                utterance_ids=batch_utt_ids,
                output_dir=adversarial_audio_dir,
            )

        utterance_ids.extend(batch_utt_ids)
        defended_scores.extend(defended_batch_scores.tolist())

    return utterance_ids, defended_scores, attack_stats


def _average_attack_stats(attack_stats: List[Dict[str, float]]) -> Dict[str, float]:
    if not attack_stats:
        return {}

    summary = {}
    keys = attack_stats[0].keys()
    for key in keys:
        summary[key] = sum(batch_stats[key] for batch_stats in attack_stats) / len(attack_stats)
    return summary


def _write_rank_payload(
    run_dir: Path,
    artifact_stem: str,
    runtime,
    payload: Dict[str, Any],
) -> Path:
    if runtime is None:
        raise ValueError("runtime is required for rank-local payload writes.")
    payload_path = rank_temp_path(run_dir, artifact_stem, runtime)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(payload_path, "w") as payload_file:
        json.dump(payload, payload_file)
        payload_file.write("\n")
    return payload_path


def _read_rank_payloads(
    run_dir: Path,
    artifact_stem: str,
    runtime,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for rank in range(runtime.world_size):
        payload_path = (run_dir / ".dist_eval") / f"{artifact_stem}.rank{rank:05d}.json"
        with open(payload_path, "r") as payload_file:
            payloads.append(json.load(payload_file))
        payload_path.unlink()
    return payloads


def _merge_attack_stats_payloads(
    gathered_payloads: Sequence[Dict[str, Any]],
    *,
    stats_key: str = "attack_stats",
) -> Dict[str, float]:
    merged_attack_stats: List[Dict[str, float]] = []
    for payload in gathered_payloads:
        if payload is None:
            continue
        attack_stats = payload.get(stats_key)
        if attack_stats is None:
            continue
        merged_attack_stats.extend(attack_stats)
    return _average_attack_stats(merged_attack_stats)


def _merge_rank_audio_dirs(
    final_audio_dir: Path,
    runtime,
) -> None:
    final_audio_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(runtime.world_size):
        source_dir = (final_audio_dir.parent / ".dist_eval") / f"{final_audio_dir.name}.rank{rank:05d}"
        if not source_dir.exists():
            continue
        for source_path in sorted(source_dir.iterdir()):
            destination_path = final_audio_dir / source_path.name
            shutil.move(str(source_path), str(destination_path))
        shutil.rmtree(source_dir)


def _device_for_runtime(runtime) -> Any:
    if runtime is not None:
        return runtime.device
    return resolve_eval_device()


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


def build_defended_metric_delta_summary(
    clean_metrics: Dict[str, float],
    adv_metrics: Dict[str, float],
    defended_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    summary = {}
    for metric_key in ("min_dcf", "eer", "cllr"):
        clean_value = clean_metrics[metric_key]
        adv_value = adv_metrics[metric_key]
        defended_value = defended_metrics[metric_key]
        summary[metric_key] = {
            "clean": clean_value,
            "adversarial": adv_value,
            "defended": defended_value,
            "absolute_delta": adv_value - clean_value,
            "relative_delta": _relative_delta(clean_value, adv_value),
            "adversarial_vs_clean_absolute_delta": adv_value - clean_value,
            "adversarial_vs_clean_relative_delta": _relative_delta(clean_value, adv_value),
            "defended_vs_clean_absolute_delta": defended_value - clean_value,
            "defended_vs_clean_relative_delta": _relative_delta(
                clean_value, defended_value
            ),
            "defended_vs_adversarial_absolute_delta": defended_value - adv_value,
            "defended_vs_adversarial_relative_delta": _relative_delta(
                adv_value, defended_value
            ),
        }
    return summary


def build_defended_adv_metric_summary(
    adv_metrics: Dict[str, float],
    defended_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    summary = {}
    for metric_key in ("min_dcf", "eer", "cllr"):
        adv_value = adv_metrics[metric_key]
        defended_value = defended_metrics[metric_key]
        summary[metric_key] = {
            "adversarial": adv_value,
            "defended": defended_value,
            "defended_vs_adversarial_absolute_delta": defended_value - adv_value,
            "defended_vs_adversarial_relative_delta": _relative_delta(
                adv_value, defended_value
            ),
        }
    return summary


def build_clean_defended_metric_summary(
    clean_metrics: Dict[str, float],
    defended_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    summary = {}
    for metric_key in ("min_dcf", "eer", "cllr"):
        clean_value = clean_metrics[metric_key]
        defended_value = defended_metrics[metric_key]
        summary[metric_key] = {
            "clean": clean_value,
            "defended": defended_value,
            "defended_vs_clean_absolute_delta": defended_value - clean_value,
            "defended_vs_clean_relative_delta": _relative_delta(
                clean_value, defended_value
            ),
        }
    return summary


def build_defended_only_metric_summary(
    defended_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        "min_dcf": {
            "defended": defended_metrics["min_dcf"],
        },
        "eer": {
            "defended": defended_metrics["eer"],
        },
        "cllr": {
            "defended": defended_metrics["cllr"],
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
    attack_name = str(summary.get("attack_name", "fgsm")).upper()

    def _format_metric_value(metric_key: str, value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        if metric_key in ("eer", "cllr"):
            return f"{value * 100:.9f}%"
        return f"{value:.9f}"

    def _format_relative_value(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.9f}%"

    with open(summary_path, "w") as summary_file:
        summary_file.write(f"{attack_name} METRIC SUMMARY\n")
        summary_file.write(f"split = {summary['split']}\n")
        summary_file.write(f"architecture = {summary['architecture']}\n")
        summary_file.write(f"checkpoint_path = {summary['checkpoint_path']}\n")
        summary_file.write(f"dataset_root = {summary['dataset_root']}\n")
        summary_file.write(f"trial_file = {summary['trial_file']}\n")
        summary_file.write(f"output_dir = {summary['output_dir']}\n")
        summary_file.write(f"device = {summary['device']}\n")
        summary_file.write(f"batch_size = {summary['batch_size']}\n")
        summary_file.write(f"defense_samples = {summary['defense_samples']}\n")
        if "defense_config_path" in summary:
            summary_file.write(f"defense_config_path = {summary['defense_config_path']}\n")
        if "defense_sigma" in summary:
            summary_file.write(f"defense_sigma = {summary['defense_sigma']}\n")
        if "defense_normalize" in summary:
            summary_file.write(
                f"defense_normalize = {summary['defense_normalize']}\n"
            )
        if "defense_clamp" in summary:
            summary_file.write(f"defense_clamp = {summary['defense_clamp']}\n")
        if "epsilon" in summary:
            summary_file.write(f"epsilon = {summary['epsilon']}\n")
        if "steps" in summary:
            summary_file.write(f"steps = {summary['steps']}\n")
        if "alpha" in summary:
            summary_file.write(f"alpha = {summary['alpha']}\n")
        if "random_start" in summary:
            summary_file.write(f"random_start = {summary['random_start']}\n")
        if "clamp_min" in summary:
            summary_file.write(f"clamp_min = {summary['clamp_min']}\n")
        if "clamp_max" in summary:
            summary_file.write(f"clamp_max = {summary['clamp_max']}\n")
        if "clean_score_path" in summary and summary["clean_score_path"] is not None:
            summary_file.write(f"clean_score_file = {summary['clean_score_path']}\n")
        if "adv_score_path" in summary and summary["adv_score_path"] is not None:
            summary_file.write(f"adversarial_score_file = {summary['adv_score_path']}\n")
        if "defended_score_path" in summary and summary["defended_score_path"] is not None:
            summary_file.write(f"defended_score_file = {summary['defended_score_path']}\n")
        if "clean_metrics_path" in summary and summary["clean_metrics_path"] is not None:
            summary_file.write(f"clean_metric_file = {summary['clean_metrics_path']}\n")
        if "adv_metrics_path" in summary and summary["adv_metrics_path"] is not None:
            summary_file.write(f"adversarial_metric_file = {summary['adv_metrics_path']}\n")
        if "defended_metrics_path" in summary and summary["defended_metrics_path"] is not None:
            summary_file.write(
                f"defended_metric_file = {summary['defended_metrics_path']}\n"
            )
        summary_file.write(f"adversarial_audio_dir = {summary['adv_audio_dir']}\n")
        summary_file.write(f"utterance_count = {summary['utterance_count']}\n")
        summary_file.write("\n")
        for metric_key in ("min_dcf", "eer", "cllr"):
            metric = summary["metrics"][metric_key]
            label = metric_names[metric_key]
            clean_value = metric.get("clean")
            adv_value = metric.get("adversarial")
            defended_value = metric.get("defended")

            if clean_value is None and adv_value is None and defended_value is not None:
                summary_file.write(
                    f"{label}: defended={_format_metric_value(metric_key, defended_value)}\n"
                )
                continue

            if clean_value is None and adv_value is not None and defended_value is None:
                adv_text = _format_metric_value(metric_key, adv_value)
                summary_file.write(f"{label}: adversarial={adv_text}\n")
                continue

            if clean_value is None and adv_value is not None:
                summary_file.write(
                    f"{label}: adversarial={_format_metric_value(metric_key, adv_value)}, "
                    f"defended={_format_metric_value(metric_key, defended_value)}, "
                    "defended_vs_adversarial_absolute_delta="
                    f"{_format_metric_value(metric_key, metric.get('defended_vs_adversarial_absolute_delta'))}, "
                    "defended_vs_adversarial_relative_delta="
                    f"{_format_relative_value(metric.get('defended_vs_adversarial_relative_delta'))}\n"
                )
                continue

            if clean_value is not None and adv_value is None and defended_value is not None:
                summary_file.write(
                    f"{label}: clean={_format_metric_value(metric_key, clean_value)}, "
                    f"defended={_format_metric_value(metric_key, defended_value)}, "
                    "defended_vs_clean_absolute_delta="
                    f"{_format_metric_value(metric_key, metric.get('defended_vs_clean_absolute_delta'))}, "
                    "defended_vs_clean_relative_delta="
                    f"{_format_relative_value(metric.get('defended_vs_clean_relative_delta'))}\n"
                )
                continue

            if clean_value is not None and defended_value is None and adv_value is not None:
                summary_file.write(
                    f"{label}: clean={_format_metric_value(metric_key, clean_value)}, "
                    f"adversarial={_format_metric_value(metric_key, adv_value)}, "
                    "absolute_delta="
                    f"{_format_metric_value(metric_key, metric.get('absolute_delta'))}, "
                    "relative_delta="
                    f"{_format_relative_value(metric.get('relative_delta'))}\n"
                )
                continue

            summary_file.write(
                f"{label}: clean={_format_metric_value(metric_key, clean_value)}, "
                f"adversarial={_format_metric_value(metric_key, adv_value)}, "
                f"defended={_format_metric_value(metric_key, defended_value)}, "
                "adversarial_vs_clean_absolute_delta="
                f"{_format_metric_value(metric_key, metric.get('adversarial_vs_clean_absolute_delta', metric.get('absolute_delta')))}, "
                "adversarial_vs_clean_relative_delta="
                f"{_format_relative_value(metric.get('adversarial_vs_clean_relative_delta', metric.get('relative_delta')))}, "
                "defended_vs_clean_absolute_delta="
                f"{_format_metric_value(metric_key, metric.get('defended_vs_clean_absolute_delta'))}, "
                "defended_vs_clean_relative_delta="
                f"{_format_relative_value(metric.get('defended_vs_clean_relative_delta'))}, "
                "defended_vs_adversarial_absolute_delta="
                f"{_format_metric_value(metric_key, metric.get('defended_vs_adversarial_absolute_delta'))}, "
                "defended_vs_adversarial_relative_delta="
                f"{_format_relative_value(metric.get('defended_vs_adversarial_relative_delta'))}\n"
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
    runtime = initialize_distributed_eval_runtime() if distributed_launch_requested() else None
    try:
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

        device = _device_for_runtime(runtime)
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
            distributed_runtime=runtime,
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
        if uses_multi_rank_distribution(runtime):
            _write_rank_payload(
                run_dir=run_dir,
                artifact_stem="clean_scores",
                runtime=runtime,
                payload={"utterance_ids": utterance_ids, "scores": scores},
            )
            distributed_barrier(runtime)
            if is_primary_rank(runtime):
                payloads = _read_rank_payloads(run_dir, "clean_scores", runtime)
                utterance_ids, scores = merge_gathered_score_payloads(
                    trial_records=trial_records,
                    gathered_payloads=payloads,
                )
        metrics = None
        persisted_score_path = None
        persisted_metrics_path = None
        if is_primary_rank(runtime):
            metrics = compute_cm_metrics_from_trial_records(
                trial_records=trial_records,
                utterance_ids=utterance_ids,
                scores=scores,
            )
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

        result = None
        if is_primary_rank(runtime):
            result = {
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
                "rank": 0 if runtime is None else runtime.rank,
                "world_size": 1 if runtime is None else runtime.world_size,
                "is_primary_rank": True,
            }
        result = broadcast_from_primary(runtime, result)
        if result is None:
            raise RuntimeError("Primary rank did not publish clean evaluation results.")
        return result
    finally:
        cleanup_distributed_eval_runtime(runtime)


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
    runtime = initialize_distributed_eval_runtime() if distributed_launch_requested() else None
    try:
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

        device = _device_for_runtime(runtime)
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
            distributed_runtime=runtime,
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

        rank_adv_audio_dir = adv_audio_dir
        if adv_audio_dir is not None and uses_multi_rank_distribution(runtime):
            rank_adv_audio_dir = rank_temp_dir(adv_audio_dir, runtime)

        clean_utterance_ids, clean_scores, _ = _collect_scores(
            data_loader=eval_loader,
            model=model,
            device=device,
            defense_kwargs=defense_kwargs,
            defense_samples=defense_samples,
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
            adversarial_audio_dir=rank_adv_audio_dir,
        )

        if clean_utterance_ids != adv_utterance_ids:
            raise ValueError("Clean and adversarial scoring produced different utterance orders.")

        if uses_multi_rank_distribution(runtime):
            _write_rank_payload(
                run_dir=run_dir,
                artifact_stem="clean_scores",
                runtime=runtime,
                payload={"utterance_ids": clean_utterance_ids, "scores": clean_scores},
            )
            _write_rank_payload(
                run_dir=run_dir,
                artifact_stem="fgsm_scores",
                runtime=runtime,
                payload={
                    "utterance_ids": adv_utterance_ids,
                    "scores": adv_scores,
                    "attack_stats": attack_stats,
                },
            )
            distributed_barrier(runtime)
            if is_primary_rank(runtime):
                clean_payloads = _read_rank_payloads(run_dir, "clean_scores", runtime)
                clean_utterance_ids, clean_scores = merge_gathered_score_payloads(
                    trial_records=trial_records,
                    gathered_payloads=clean_payloads,
                )
                adv_payloads = _read_rank_payloads(run_dir, "fgsm_scores", runtime)
                adv_utterance_ids, adv_scores = merge_gathered_score_payloads(
                    trial_records=trial_records,
                    gathered_payloads=adv_payloads,
                )
                attack_stats = _merge_attack_stats_payloads(adv_payloads)
                if adv_audio_dir is not None:
                    _merge_rank_audio_dirs(adv_audio_dir, runtime)
        else:
            attack_stats = _average_attack_stats(attack_stats)

        clean_metrics = None
        adv_metrics = None
        metric_summary = None
        summary = None
        persisted_clean_score_path = None
        persisted_adv_score_path = None
        persisted_clean_metrics_path = None
        persisted_adv_metrics_path = None
        persisted_summary_json_path = None
        persisted_summary_text_path = None
        persisted_adv_audio_dir = adv_audio_dir

        if is_primary_rank(runtime):
            clean_metrics = compute_cm_metrics_from_trial_records(
                trial_records=trial_records,
                utterance_ids=clean_utterance_ids,
                scores=clean_scores,
            )
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
                "attack_stats": attack_stats,
            }

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

        result = None
        if is_primary_rank(runtime):
            result = {
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
                "attack_stats": attack_stats,
                "metrics": metric_summary,
                "artifact_warnings": artifact_warnings,
                "rank": 0 if runtime is None else runtime.rank,
                "world_size": 1 if runtime is None else runtime.world_size,
                "is_primary_rank": True,
            }
        result = broadcast_from_primary(runtime, result)
        if result is None:
            raise RuntimeError("Primary rank did not publish FGSM evaluation results.")
        return result
    finally:
        cleanup_distributed_eval_runtime(runtime)


def run_pgd_scoring_pipeline(
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
    steps: int = 5,
    alpha: Optional[float] = None,
    random_start: bool = False,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    clean_score_filename: str = "clean_scores.txt",
    save_adv_audio: bool = False,
    metrics_only: bool = False,
    skip_clean_pass: bool = False,
    seed: int = 1234,
) -> Dict[str, Any]:
    from src.pgd_utils import (
        PgdAttackConfig,
        build_pgd_artifact_contract,
        build_pgd_summary_stub,
    )
    runtime = initialize_distributed_eval_runtime() if distributed_launch_requested() else None
    try:
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

        device = _device_for_runtime(runtime)
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
            distributed_runtime=runtime,
        )

        run_dir = paths.output_dir / config["model_tag"] / f"{split}_pgd_eval"
        attack_config = PgdAttackConfig(
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            random_start=random_start,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            save_adv_audio=save_adv_audio,
        )
        artifact_warnings: List[str] = []

        if metrics_only and attack_config.save_adv_audio:
            message = (
                "Ignoring save_adv_audio because metrics_only=True disables run artifacts."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            artifact_warnings.append(message)
            attack_config = PgdAttackConfig(
                epsilon=attack_config.epsilon,
                steps=attack_config.steps,
                alpha=attack_config.alpha,
                random_start=attack_config.random_start,
                clamp_min=attack_config.clamp_min,
                clamp_max=attack_config.clamp_max,
                save_adv_audio=False,
            )

        artifacts = build_pgd_artifact_contract(
            run_dir=run_dir,
            attack_config=attack_config,
            clean_score_filename=clean_score_filename,
        )
        rank_adv_audio_dir = artifacts.adv_audio_dir
        if artifacts.adv_audio_dir is not None and uses_multi_rank_distribution(runtime):
            rank_adv_audio_dir = rank_temp_dir(artifacts.adv_audio_dir, runtime)

        clean_utterance_ids = None
        clean_scores = None
        if not skip_clean_pass:
            clean_utterance_ids, clean_scores, _ = _collect_scores(
                data_loader=eval_loader,
                model=model,
                device=device,
            )

        attacked_utterance_ids, defended_scores, attack_stats = _collect_pgd_attack_scores(
            data_loader=eval_loader,
            model=model,
            device=device,
            epsilon=attack_config.epsilon,
            steps=attack_config.steps,
            alpha=attack_config.alpha,
            random_start=attack_config.random_start,
            clamp_min=attack_config.clamp_min,
            clamp_max=attack_config.clamp_max,
            defense_kwargs=defense_kwargs,
            defense_samples=defense_samples,
            adversarial_audio_dir=rank_adv_audio_dir,
        )

        if clean_utterance_ids is not None and clean_utterance_ids != attacked_utterance_ids:
            raise ValueError("Clean and attacked scoring produced different utterance orders.")

        if uses_multi_rank_distribution(runtime):
            if clean_utterance_ids is not None and clean_scores is not None:
                _write_rank_payload(
                    run_dir=run_dir,
                    artifact_stem="clean_scores",
                    runtime=runtime,
                    payload={"utterance_ids": clean_utterance_ids, "scores": clean_scores},
                )
            _write_rank_payload(
                run_dir=run_dir,
                artifact_stem="pgd_scores",
                runtime=runtime,
                payload={
                    "utterance_ids": attacked_utterance_ids,
                    "defended_scores": defended_scores,
                    "attack_stats": attack_stats,
                },
            )
            distributed_barrier(runtime)
            if is_primary_rank(runtime):
                if clean_utterance_ids is not None and clean_scores is not None:
                    clean_payloads = _read_rank_payloads(run_dir, "clean_scores", runtime)
                    clean_utterance_ids, clean_scores = merge_gathered_score_payloads(
                        trial_records=trial_records,
                        gathered_payloads=clean_payloads,
                    )
                attack_payloads = _read_rank_payloads(run_dir, "pgd_scores", runtime)
                attacked_utterance_ids, defended_scores = merge_gathered_score_payloads(
                    trial_records=trial_records,
                    gathered_payloads=attack_payloads,
                    score_key="defended_scores",
                )
                attack_stats = _merge_attack_stats_payloads(attack_payloads)
                if artifacts.adv_audio_dir is not None:
                    _merge_rank_audio_dirs(artifacts.adv_audio_dir, runtime)
        else:
            attack_stats = _average_attack_stats(attack_stats)

        clean_metrics = None
        defended_metrics = None
        metric_summary = None
        summary = None
        persisted_clean_score_path = None
        persisted_defended_score_path = None
        persisted_clean_metrics_path = None
        persisted_defended_metrics_path = None
        persisted_summary_json_path = None
        persisted_summary_text_path = None
        persisted_adv_audio_dir = artifacts.adv_audio_dir

        if is_primary_rank(runtime):
            if clean_utterance_ids is not None and clean_scores is not None:
                clean_metrics = compute_cm_metrics_from_trial_records(
                    trial_records=trial_records,
                    utterance_ids=clean_utterance_ids,
                    scores=clean_scores,
                )
            defended_metrics = compute_cm_metrics_from_trial_records(
                trial_records=trial_records,
                utterance_ids=attacked_utterance_ids,
                scores=defended_scores,
            )
            metric_summary = (
                build_defended_only_metric_summary(defended_metrics)
                if clean_metrics is None
                else build_clean_defended_metric_summary(clean_metrics, defended_metrics)
            )
            summary = build_pgd_summary_stub(
                split=split,
                architecture=config["model_config"]["architecture"],
                checkpoint_path=paths.model_weights_path,
                dataset_root=paths.dataset_root,
                trial_file=trial_path,
                output_dir=run_dir,
                device=str(device),
                batch_size=effective_batch_size,
                attack_config=attack_config,
                artifacts=artifacts,
            )
            summary["utterance_count"] = len(attacked_utterance_ids)
            summary["metrics"] = metric_summary
            summary["attack_stats"] = attack_stats
            summary["defense_config_path"] = config.get("defense_config_path")
            summary["defense_sigma"] = defense_kwargs["sigma"]
            summary["defense_normalize"] = defense_kwargs["normalize"]
            summary["defense_clamp"] = defense_kwargs["clamp"]
            summary["defense_samples"] = defense_samples
            if skip_clean_pass:
                summary["clean_score_path"] = None
                summary["clean_metrics_path"] = None
                summary["clean_pass_skipped"] = True

            if not metrics_only:
                if not skip_clean_pass and clean_utterance_ids is not None and clean_scores is not None:
                    if _try_write_artifact(
                        action=lambda: _write_ordered_score_lines(
                            save_path=artifacts.clean_score_path,
                            trial_records=trial_records,
                            utterance_ids=clean_utterance_ids,
                            scores=clean_scores,
                        ),
                        description=f"clean score file '{artifacts.clean_score_path}'",
                        artifact_warnings=artifact_warnings,
                    ):
                        persisted_clean_score_path = artifacts.clean_score_path

                if _try_write_artifact(
                    action=lambda: _write_ordered_score_lines(
                        save_path=artifacts.defended_score_path,
                        trial_records=trial_records,
                        utterance_ids=attacked_utterance_ids,
                        scores=defended_scores,
                    ),
                    description=f"defended adversarial score file '{artifacts.defended_score_path}'",
                    artifact_warnings=artifact_warnings,
                ):
                    persisted_defended_score_path = artifacts.defended_score_path

                if not skip_clean_pass and clean_metrics is not None:
                    if _try_write_artifact(
                        action=lambda: write_metric_report(
                            metrics_path=artifacts.clean_metrics_path,
                            min_dcf=clean_metrics["min_dcf"],
                            eer=clean_metrics["eer"],
                            cllr=clean_metrics["cllr"],
                        ),
                        description=f"clean metric report '{artifacts.clean_metrics_path}'",
                        artifact_warnings=artifact_warnings,
                    ):
                        persisted_clean_metrics_path = artifacts.clean_metrics_path

                if _try_write_artifact(
                    action=lambda: write_metric_report(
                        metrics_path=artifacts.defended_metrics_path,
                        min_dcf=defended_metrics["min_dcf"],
                        eer=defended_metrics["eer"],
                        cllr=defended_metrics["cllr"],
                    ),
                    description=f"defended adversarial metric report '{artifacts.defended_metrics_path}'",
                    artifact_warnings=artifact_warnings,
                ):
                    persisted_defended_metrics_path = artifacts.defended_metrics_path

                if _try_write_artifact(
                    action=lambda: write_metric_summary_json(artifacts.summary_json_path, summary),
                    description=f"PGD summary JSON '{artifacts.summary_json_path}'",
                    artifact_warnings=artifact_warnings,
                ):
                    persisted_summary_json_path = artifacts.summary_json_path

                if _try_write_artifact(
                    action=lambda: write_metric_summary_text(artifacts.summary_text_path, summary),
                    description=f"PGD summary text '{artifacts.summary_text_path}'",
                    artifact_warnings=artifact_warnings,
                ):
                    persisted_summary_text_path = artifacts.summary_text_path
            else:
                persisted_adv_audio_dir = None

        result = None
        if is_primary_rank(runtime):
            result = {
                "split": split,
                "device": str(device),
                "weights_path": paths.model_weights_path,
                "trial_path": trial_path,
                "batch_size": effective_batch_size,
                "defense_config_path": config.get("defense_config_path"),
                "defense_sigma": defense_kwargs["sigma"],
                "defense_normalize": defense_kwargs["normalize"],
                "defense_clamp": defense_kwargs["clamp"],
                "defense_samples": defense_samples,
                "epsilon": attack_config.epsilon,
                "steps": attack_config.steps,
                "alpha": attack_config.resolved_alpha,
                "random_start": attack_config.random_start,
                "clamp_min": attack_config.clamp_min,
                "clamp_max": attack_config.clamp_max,
                "clean_score_path": persisted_clean_score_path,
                "adv_score_path": None,
                "defended_score_path": persisted_defended_score_path,
                "clean_metrics_path": persisted_clean_metrics_path,
                "adv_metrics_path": None,
                "defended_metrics_path": persisted_defended_metrics_path,
                "adv_audio_dir": persisted_adv_audio_dir,
                "summary_json_path": persisted_summary_json_path,
                "summary_text_path": persisted_summary_text_path,
                "utterance_count": len(attacked_utterance_ids),
                "attack_stats": attack_stats,
                "metrics": metric_summary,
                "artifact_warnings": artifact_warnings,
                "skip_clean_pass": skip_clean_pass,
                "rank": 0 if runtime is None else runtime.rank,
                "world_size": 1 if runtime is None else runtime.world_size,
                "is_primary_rank": True,
            }
        result = broadcast_from_primary(runtime, result)
        if result is None:
            raise RuntimeError("Primary rank did not publish PGD evaluation results.")
        return result
    finally:
        cleanup_distributed_eval_runtime(runtime)
