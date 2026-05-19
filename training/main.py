"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchcontrib.optim import SWA

from src.data_utils import (TrainDataset,TestDataset, genSpoof_list, load_trial_records)
from src.defense_utils import (
    apply_resolved_defense_config,
    forward_with_defense,
    get_defense_kwargs,
    get_defense_samples,
)
from src.eval_utils import (
    canonicalize_scores_by_trial_records,
    load_plain_model_weights,
    merge_gathered_score_payloads,
)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from src.path_utils import WorkflowPaths, apply_path_overrides, resolve_workflow_paths
from src.utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


@dataclass(frozen=True)
class DistributedRuntime:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


@dataclass
class LoaderBundle:
    train_loader: Optional[DataLoader]
    dev_loader: Optional[DataLoader]
    eval_loader: Optional[DataLoader]
    train_sampler: Optional[DistributedSampler]
    dev_sampler: Optional[DistributedSampler]
    eval_sampler: Optional[DistributedSampler]


def load_training_config(
    config_path: str,
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    ssl_pretrained_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
    musan_path: Optional[str] = None,
    rir_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        config = json.load(file)

    if dataset_root is not None:
        config["database_path"] = dataset_root
    if metadata_root is not None:
        config["metadata_path"] = metadata_root
    if ssl_pretrained_path is not None:
        config["model_config"]["ssl_pretrained_path"] = ssl_pretrained_path
    if musan_path is not None:
        config["musan_path"] = musan_path
    if rir_path is not None:
        config["rir_path"] = rir_path
    if batch_size is not None:
        config["batch_size"] = batch_size

    return apply_resolved_defense_config(
        config=config,
        config_path=config_path,
        defense_config_path=defense_config_path,
    )


def apply_training_path_fallbacks(
    config: Dict[str, Any],
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    train_trial_file: Optional[str] = None,
    dev_trial_file: Optional[str] = None,
    eval_trial_file: Optional[str] = None,
    train_audio_root: Optional[str] = None,
    dev_audio_root: Optional[str] = None,
    eval_audio_root: Optional[str] = None,
) -> Dict[str, Any]:
    if dataset_root is None:
        inferred_dataset_root = _infer_dataset_root_from_audio_roots(
            train_audio_root=train_audio_root,
            dev_audio_root=dev_audio_root,
            eval_audio_root=eval_audio_root,
        )
        if inferred_dataset_root is not None:
            config["database_path"] = inferred_dataset_root

    if metadata_root is None:
        for trial_file in (train_trial_file, dev_trial_file, eval_trial_file):
            if trial_file is not None:
                config["metadata_path"] = str(
                    Path(trial_file).expanduser().resolve(strict=False).parent
                )
                break

    return config


def _infer_dataset_root_from_audio_roots(
    train_audio_root: Optional[str] = None,
    dev_audio_root: Optional[str] = None,
    eval_audio_root: Optional[str] = None,
) -> Optional[str]:
    for raw_path in (train_audio_root, dev_audio_root, eval_audio_root):
        if raw_path is None:
            continue
        audio_root = Path(raw_path).expanduser().resolve(strict=False)
        if audio_root.name in {"flac_T", "flac_D", "flac_E"}:
            return str(audio_root.parent)
        if audio_root.name == "flac_E_eval" and audio_root.parent.name == "eval_full":
            return str(audio_root.parent.parent)
        return str(audio_root.parent)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Configuration file.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        help="Output directory for results.",
        default="./exp_result",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--metadata-root",
        "--metadata_root",
        dest="metadata_root",
        default=None,
        help="Optional metadata root override.",
    )
    parser.add_argument(
        "--train-trial-file",
        "--train_trial_file",
        dest="train_trial_file",
        default=None,
        help="Optional explicit train TSV override.",
    )
    parser.add_argument(
        "--dev-trial-file",
        "--dev_trial_file",
        dest="dev_trial_file",
        default=None,
        help="Optional explicit dev TSV override.",
    )
    parser.add_argument(
        "--eval-trial-file",
        "--eval_trial_file",
        dest="eval_trial_file",
        default=None,
        help="Optional explicit eval TSV override.",
    )
    parser.add_argument(
        "--train-audio-root",
        "--train_audio_root",
        dest="train_audio_root",
        default=None,
        help="Optional explicit train audio directory override.",
    )
    parser.add_argument(
        "--dev-audio-root",
        "--dev_audio_root",
        dest="dev_audio_root",
        default=None,
        help="Optional explicit dev audio directory override.",
    )
    parser.add_argument(
        "--eval-audio-root",
        "--eval_audio_root",
        dest="eval_audio_root",
        default=None,
        help="Optional explicit eval audio directory override.",
    )
    parser.add_argument(
        "--ssl-pretrained-path",
        "--ssl_pretrained_path",
        dest="ssl_pretrained_path",
        default=None,
        help="Optional WavLM checkpoint override.",
    )
    parser.add_argument(
        "--defense-config",
        "--defense_config",
        dest="defense_config",
        default=None,
        help="Optional shared defense config JSON override.",
    )
    parser.add_argument(
        "--musan-path",
        "--musan_path",
        dest="musan_path",
        default=None,
        help="Optional MUSAN directory override.",
    )
    parser.add_argument(
        "--rir-path",
        "--rir_path",
        dest="rir_path",
        default=None,
        help="Optional RIR directory override.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--dev-batch-size",
        "--dev_batch_size",
        dest="dev_batch_size",
        type=int,
        default=None,
        help="Optional dev batch size override. Defaults to training batch size if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--start-val-epoch",
        "--start_val_epoch",
        dest="start_val_epoch",
        type=int,
        default=10,
        help="Epoch index to start validation.",
    )
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Run model training.",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Skip model training.",
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="Run final evaluation on the eval split.",
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Skip final evaluation on the eval split.",
    )
    parser.set_defaults(train=True, eval=False)
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Comment to describe the saved model.",
    )
    parser.add_argument(
        "--eval-model-weights",
        "--eval_model_weights",
        dest="eval_model_weights",
        type=str,
        default=None,
        help="Path to the model weight file for evaluation.",
    )
    return parser.parse_args()


def _parse_distributed_env_int(name: str, value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got '{value}'."
        ) from exc
    return parsed_value


def initialize_distributed_runtime() -> DistributedRuntime:
    env_values = {
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "RANK": os.environ.get("RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }
    provided_values = {name: value for name, value in env_values.items() if value is not None}
    if not provided_values:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedRuntime(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    missing_names = [name for name, value in env_values.items() if value is None]
    if missing_names:
        raise ValueError(
            "Distributed launch environment is incomplete. Expected LOCAL_RANK, "
            f"RANK, and WORLD_SIZE together, but missing: {', '.join(missing_names)}."
        )

    local_rank = _parse_distributed_env_int("LOCAL_RANK", env_values["LOCAL_RANK"])
    rank = _parse_distributed_env_int("RANK", env_values["RANK"])
    world_size = _parse_distributed_env_int("WORLD_SIZE", env_values["WORLD_SIZE"])

    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}.")
    if rank < 0:
        raise ValueError(f"RANK must be non-negative, got {rank}.")
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be at least 1, got {world_size}.")
    if rank >= world_size:
        raise ValueError(
            f"RANK must be smaller than WORLD_SIZE, got rank={rank}, world_size={world_size}."
        )

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError(
            f"Distributed execution requested with WORLD_SIZE={world_size}, "
            "but no CUDA GPUs are visible."
        )
    if visible_gpu_count < world_size:
        raise RuntimeError(
            "Distributed execution requested with WORLD_SIZE="
            f"{world_size}, but only {visible_gpu_count} visible CUDA GPU(s) were detected."
        )
    if local_rank >= visible_gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for {visible_gpu_count} visible CUDA GPU(s)."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return DistributedRuntime(
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed_runtime(runtime: DistributedRuntime) -> None:
    if runtime.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_primary_rank(runtime: DistributedRuntime) -> bool:
    return runtime.rank == 0


def distributed_barrier(runtime: DistributedRuntime) -> None:
    if runtime.is_distributed:
        dist.barrier()


def uses_multi_rank_distribution(runtime: Optional[DistributedRuntime]) -> bool:
    return runtime is not None and runtime.is_distributed and runtime.world_size > 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def broadcast_from_primary(
    runtime: DistributedRuntime,
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not runtime.is_distributed:
        return payload
    payload_container = [payload if is_primary_rank(runtime) else None]
    dist.broadcast_object_list(payload_container, src=0)
    return payload_container[0]


def load_plain_checkpoint(
    model: torch.nn.Module,
    model_path: Path,
    device: torch.device,
) -> None:
    load_plain_model_weights(unwrap_model(model), model_path, device)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    if not args.train and not args.eval:
        raise ValueError("Nothing to do: enable training and/or evaluation.")

    runtime = initialize_distributed_runtime()
    writer: Optional[SummaryWriter] = None
    f_log = None
    try:
        # load experiment configurations
        config = load_training_config(
            config_path=args.config,
            dataset_root=args.dataset_root,
            metadata_root=args.metadata_root,
            ssl_pretrained_path=args.ssl_pretrained_path,
            defense_config_path=args.defense_config,
            musan_path=args.musan_path,
            rir_path=args.rir_path,
            batch_size=args.batch_size,
        )
        config = apply_training_path_fallbacks(
            config=config,
            dataset_root=args.dataset_root,
            metadata_root=args.metadata_root,
            train_trial_file=args.train_trial_file,
            dev_trial_file=args.dev_trial_file,
            eval_trial_file=args.eval_trial_file,
            train_audio_root=args.train_audio_root,
            dev_audio_root=args.dev_audio_root,
            eval_audio_root=args.eval_audio_root,
        )
        workflow_paths = resolve_workflow_paths(
            config=config,
            output_dir=args.output_dir,
            model_weights_path=args.eval_model_weights,
            require_training_assets=args.train,
            require_dev_assets=args.train,
            require_eval_assets=args.eval,
            train_audio_root_override=args.train_audio_root,
            dev_audio_root_override=args.dev_audio_root,
            eval_audio_root_override=args.eval_audio_root,
            train_metadata_override=args.train_trial_file,
            dev_metadata_override=args.dev_trial_file,
            eval_metadata_override=args.eval_trial_file,
        )
        config = apply_path_overrides(config, workflow_paths)
        model_config = config["model_config"]
        optim_config = config["optim_config"]
        optim_config["epochs"] = config["num_epochs"]
        if "freq_aug" not in config:
            config["freq_aug"] = "False"

        # make experiment reproducible
        set_seed(args.seed, config)

        # define database related paths
        output_dir = workflow_paths.output_dir
        dev_trial_path = workflow_paths.dev_metadata
        # define model related paths
        model_tag = config["model_tag"]
        model_tag = output_dir / model_tag
        model_save_path = model_tag / "weights"
        eval_score_path = model_tag / config["eval_output"]
        if is_primary_rank(runtime):
            writer = SummaryWriter(model_tag)
            os.makedirs(model_save_path, exist_ok=True)
            copy(args.config, model_tag / "config.conf")
        distributed_barrier(runtime)

        device = runtime.device
        print(
            "Device: {}{}".format(
                device,
                (
                    f" | distributed rank {runtime.rank}/{runtime.world_size} "
                    f"(local_rank={runtime.local_rank})"
                )
                if runtime.is_distributed
                else "",
            )
        )
        defense_kwargs = get_defense_kwargs(config)
        defense_samples = get_defense_samples(config)

        # define model architecture
        model = get_model(model_config, device)
        if runtime.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        # define dataloaders
        loaders = get_loader(
            workflow_paths,
            args.seed,
            config,
            load_train=args.train,
            load_dev=args.train,
            load_eval=args.eval,
            distributed_runtime=runtime,
            distributed_splits=(
                {"train", "dev", "eval"}
                if uses_multi_rank_distribution(runtime)
                else set()
            ),
            dev_batch_size=args.dev_batch_size,
        )
        trn_loader = loaders.train_loader
        dev_loader = loaders.dev_loader
        eval_loader = loaders.eval_loader

        optimizer = None
        scheduler = None
        optimizer_swa = None
        if args.train:
            optim_config["steps_per_epoch"] = len(trn_loader)
            optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
            optimizer_swa = SWA(optimizer)

        best_dev_eer = 100.
        best_dev_dcf = 1.
        best_dev_cllr = 1.
        no_improve = 0  # number of snapshots of model to use in SWA
        if is_primary_rank(runtime):
            f_log = open(model_tag / "metric_log.txt", "a")
            f_log.write("=" * 5 + "\n")
            f_log.flush()

        # make directory for metric logging
        metric_path = model_tag / "metrics"
        if is_primary_rank(runtime):
            os.makedirs(metric_path, exist_ok=True)
        distributed_barrier(runtime)

        if args.train:
            # Training
            for epoch in range(config["num_epochs"]):
                print("training epoch{:03d}".format(epoch))
                if loaders.train_sampler is not None:
                    loaders.train_sampler.set_epoch(epoch)

                running_loss = train_epoch(trn_loader, model, optimizer, device,
                                           scheduler, config)
                epoch_checkpoint_path = None
                if is_primary_rank(runtime):
                    epoch_checkpoint_path = save_epoch_checkpoint(
                        model=model,
                        model_save_path=model_save_path,
                        epoch=epoch,
                    )
                    print("Saved checkpoint: {}".format(epoch_checkpoint_path))
                if args.start_val_epoch == 0 or epoch % args.start_val_epoch != 0:
                    print("DONE.\nLoss:{:.5f}. Skip validation step".format(running_loss))
                    distributed_barrier(runtime)
                    continue

                print(f"[rank {runtime.rank}] Starting produce_evaluation_file...", flush=True)
                validation_state = None
                produce_evaluation_file(
                    dev_loader,
                    model,
                    device,
                    metric_path/"dev_score.txt",
                    dev_trial_path,
                    runtime=runtime,
                    defense_kwargs=defense_kwargs,
                    defense_samples=defense_samples,
                )
                print(f"[rank {runtime.rank}] produce_evaluation_file done.", flush=True)

                if is_primary_rank(runtime):
                    print("[rank 0] Calculating metrics...", flush=True)
                    dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
                        cm_scores_file=metric_path/"dev_score.txt",
                        output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
                        printout=False)
                    print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
                        running_loss, dev_eer, dev_dcf, dev_cllr))
                    writer.add_scalar("loss", running_loss, epoch)
                    writer.add_scalar("dev_eer", dev_eer, epoch)
                    writer.add_scalar("dev_dcf", dev_dcf, epoch)
                    writer.add_scalar("dev_cllr", dev_cllr, epoch)

                    best_dev_dcf = min(dev_dcf, best_dev_dcf)
                    best_dev_cllr = min(dev_cllr, best_dev_cllr)
                    is_best_model = best_dev_eer >= dev_eer
                    if is_best_model:
                        print("best model find at epoch", epoch)
                        best_dev_eer = dev_eer
                        best_model_link = model_save_path / "best_model.pth"
                        if best_model_link.is_symlink() or best_model_link.exists():
                            best_model_link.unlink()
                        best_model_link.symlink_to(epoch_checkpoint_path.name)
                        print("Saving epoch {} for swa".format(epoch))
                        no_improve = 0
                    else:
                        no_improve += 1
                    writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
                    writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
                    writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)
                    if f_log is not None:
                        f_log.write(
                            "epoch={:03d}, loss={:.5f}, dev_eer={:.3f}, dev_dcf={:.5f}, "
                            "dev_cllr={:.5f}, best_dev_eer={:.3f}, no_improve={}\n".format(
                                epoch,
                                running_loss,
                                dev_eer,
                                dev_dcf,
                                dev_cllr,
                                best_dev_eer,
                                no_improve,
                            )
                        )
                        f_log.flush()
                    validation_state = {
                        "best_dev_eer": best_dev_eer,
                        "best_dev_dcf": best_dev_dcf,
                        "best_dev_cllr": best_dev_cllr,
                        "no_improve": no_improve,
                        "is_best_model": is_best_model,
                    }
                    print("[rank 0] Validation state ready, reaching barrier...", flush=True)

                print(f"[rank {runtime.rank}] Reaching barrier before broadcast...", flush=True)
                distributed_barrier(runtime)
                print(f"[rank {runtime.rank}] Past barrier, broadcasting...", flush=True)
                validation_state = broadcast_from_primary(runtime, validation_state)
                print(f"[rank {runtime.rank}] Broadcast done.", flush=True)
                if validation_state is None:
                    raise RuntimeError("Primary rank did not publish validation state.")

                best_dev_eer = validation_state["best_dev_eer"]
                best_dev_dcf = validation_state["best_dev_dcf"]
                best_dev_cllr = validation_state["best_dev_cllr"]
                no_improve = validation_state["no_improve"]
                if validation_state["is_best_model"]:
                    print(f"[rank {runtime.rank}] Updating SWA...", flush=True)
                    optimizer_swa.update_swa()
                if no_improve >= config["early_stop_epochs"]:
                    print(f"[rank {runtime.rank}] Early stopping triggered.", flush=True)
                    break

        # evaluates pretrained model on the evaluation split
        if args.eval:
            distributed_barrier(runtime)
            eval_trial_path = workflow_paths.eval_metadata
            model_path = workflow_paths.model_weights_path or (model_save_path / "best_model.pth")
            load_plain_checkpoint(model, model_path, device)
            if is_primary_rank(runtime):
                print("Model loaded : {}".format(model_path))
                print("Start evaluation...")
            produce_evaluation_file(
                eval_loader,
                model,
                device,
                eval_score_path,
                eval_trial_path,
                runtime=runtime,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
            )
            if is_primary_rank(runtime):
                eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
                    cm_scores_file=eval_score_path,
                    output_file=model_tag/"loaded_model_result.txt")
                print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))
            distributed_barrier(runtime)
            return
    finally:
        if writer is not None:
            writer.close()
        if f_log is not None:
            f_log.close()
        cleanup_distributed_runtime(runtime)


def save_epoch_checkpoint(
    model: torch.nn.Module,
    model_save_path: Path,
    epoch: int,
) -> Path:
    checkpoint_path = model_save_path / "epoch_{:03d}.pth".format(epoch)
    torch.save(unwrap_model(model).state_dict(), checkpoint_path)
    return checkpoint_path

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config, device=device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def _coerce_loader_bool(value: Any, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str_to_bool(value)
    raise ValueError(
        f"Loader option '{option_name}' must be a boolean or boolean-like string."
    )


def _coerce_loader_int(
    value: Any,
    option_name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Loader option '{option_name}' must be an integer.")
    if value < minimum:
        raise ValueError(
            f"Loader option '{option_name}' must be greater than or equal to {minimum}."
        )
    return value


def _get_loader_settings(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    loader_config = config.get("loader_config")
    if loader_config is None:
        loader_config = {}
    if not isinstance(loader_config, dict):
        raise ValueError("config['loader_config'] must be a JSON object when provided.")

    split_config = loader_config.get(split)
    if split_config is None:
        split_config = {}
    if not isinstance(split_config, dict):
        raise ValueError(
            f"config['loader_config']['{split}'] must be a JSON object when provided."
        )

    num_workers_value = split_config.get(
        "num_workers",
        loader_config.get("num_workers", 0),
    )
    num_workers = _coerce_loader_int(
        num_workers_value,
        f"loader_config.{split}.num_workers",
    )

    settings: Dict[str, Any] = {"num_workers": num_workers}
    if num_workers == 0:
        return settings

    persistent_value = split_config.get(
        "persistent_workers",
        loader_config.get("persistent_workers"),
    )
    if persistent_value is not None:
        settings["persistent_workers"] = _coerce_loader_bool(
            persistent_value,
            f"loader_config.{split}.persistent_workers",
        )

    prefetch_value = split_config.get(
        "prefetch_factor",
        loader_config.get("prefetch_factor"),
    )
    if prefetch_value is not None:
        settings["prefetch_factor"] = _coerce_loader_int(
            prefetch_value,
            f"loader_config.{split}.prefetch_factor",
            minimum=1,
        )

    return settings


def _build_dataloader_kwargs(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    loader_settings = _get_loader_settings(config, split)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": loader_settings["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_settings["num_workers"] > 0:
        if "persistent_workers" in loader_settings:
            loader_kwargs["persistent_workers"] = loader_settings["persistent_workers"]
        if "prefetch_factor" in loader_settings:
            loader_kwargs["prefetch_factor"] = loader_settings["prefetch_factor"]
    return loader_kwargs


def get_loader(
        workflow_paths: WorkflowPaths,
        seed: int,
        config: dict,
        distributed_runtime: Optional[DistributedRuntime] = None,
        distributed_splits: Optional[Set[str]] = None,
        load_train: bool = True,
        load_dev: bool = True,
        load_eval: bool = True,
        dev_batch_size: Optional[int] = None) -> LoaderBundle:
    """Make PyTorch DataLoaders for train / development / evaluation."""

    trn_loader = None
    dev_loader = None
    eval_loader = None
    train_sampler = None
    dev_sampler = None
    eval_sampler = None
    if distributed_splits is None:
        distributed_splits = {"train"} if distributed_runtime and distributed_runtime.is_distributed else set()

    if load_train:
        trn_database_path = workflow_paths.train_audio_root
        trn_list_path = workflow_paths.train_metadata
        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path)
        print("no. training files:", len(file_train))

        train_set = TrainDataset(list_IDs=file_train,
                                               labels=d_label_trn,
                                               base_dir=trn_database_path,
                                               add_noise=str_to_bool(config["add_noise"]),
                                               musan_path=config.get("musan_path", "musan_data"),
                                               rir_path=config.get("rir_path", "RIR_data"))
        gen = torch.Generator()
        gen.manual_seed(seed)
        if distributed_runtime and distributed_runtime.is_distributed and "train" in distributed_splits:
            train_sampler = DistributedSampler(
                train_set,
                num_replicas=distributed_runtime.world_size,
                rank=distributed_runtime.rank,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
        train_loader_kwargs = _build_dataloader_kwargs(config, "train")
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=train_sampler is None,
                                sampler=train_sampler,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=gen,
                                **train_loader_kwargs)

    if load_dev:
        dev_database_path = workflow_paths.dev_audio_root
        dev_trial_path = workflow_paths.dev_metadata
        _, file_dev = genSpoof_list(dir_meta=dev_trial_path)
        print("no. dev files:", len(file_dev))

        dev_set = TestDataset(list_IDs=file_dev,
                                                base_dir=dev_database_path)
        if distributed_runtime and distributed_runtime.is_distributed and "dev" in distributed_splits:
            dev_sampler = DistributedSampler(
                dev_set,
                num_replicas=distributed_runtime.world_size,
                rank=distributed_runtime.rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
        dev_loader_kwargs = _build_dataloader_kwargs(config, "dev")
        dev_loader = DataLoader(dev_set,
                                batch_size=dev_batch_size or config["batch_size"],
                                shuffle=False,
                                sampler=dev_sampler,
                                drop_last=False,
                                **dev_loader_kwargs)

    if load_eval:
        eval_database_path = workflow_paths.eval_audio_root
        eval_trial_path = workflow_paths.eval_metadata
        _, file_eval = genSpoof_list(dir_meta=eval_trial_path)
        print("no. evaluation files:", len(file_eval))

        eval_set = TestDataset(list_IDs=file_eval,
                                                base_dir=eval_database_path)
        if distributed_runtime and distributed_runtime.is_distributed and "eval" in distributed_splits:
            eval_sampler = DistributedSampler(
                eval_set,
                num_replicas=distributed_runtime.world_size,
                rank=distributed_runtime.rank,
                shuffle=False,
                seed=seed,
                drop_last=False,
            )
        eval_loader_kwargs = _build_dataloader_kwargs(config, "eval")
        eval_loader = DataLoader(eval_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                sampler=eval_sampler,
                                drop_last=False,
                                **eval_loader_kwargs)

    return LoaderBundle(
        train_loader=trn_loader,
        dev_loader=dev_loader,
        eval_loader=eval_loader,
        train_sampler=train_sampler,
        dev_sampler=dev_sampler,
        eval_sampler=eval_sampler,
    )

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    runtime: Optional[DistributedRuntime] = None,
    defense_kwargs: Optional[Dict[str, Any]] = None,
    defense_samples: int = 1) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    defense_kwargs = defense_kwargs or {}
    trial_records = load_trial_records(trial_path)
    fname_list = []
    score_list = []
    use_non_blocking = device.type == "cuda"
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=use_non_blocking)
        with torch.no_grad():
            batch_out, _ = forward_with_defense(
                model=model,
                wav=batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
                vectorized=defense_samples > 1,
            )
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    if uses_multi_rank_distribution(runtime):
        print(f"[rank {runtime.rank}] Writing temp score file...", flush=True)
        tmp_path = Path(save_path).parent / f"tmp_scores_rank{runtime.rank}.txt"
        with open(tmp_path, "w") as f:
            for fn, sc in zip(fname_list, score_list):
                f.write(f"{fn} {sc}\n")
        print(f"[rank {runtime.rank}] Temp file written. Waiting at barrier...", flush=True)

        distributed_barrier(runtime)

        if not is_primary_rank(runtime):
            print(f"[rank {runtime.rank}] Past barrier, returning.", flush=True)
            return

        print(f"[rank 0] All ranks done. Merging temp files...", flush=True)
        all_fnames = []
        all_scores = []
        for r in range(runtime.world_size):
            rpath = Path(save_path).parent / f"tmp_scores_rank{r}.txt"
            with open(rpath) as f:
                for line in f:
                    fn, sc = line.strip().split()
                    all_fnames.append(fn)
                    all_scores.append(float(sc))
            rpath.unlink()
            print(f"[rank 0] Merged rank {r} scores.", flush=True)

        fname_list = all_fnames
        score_list = all_scores
        print(f"[rank 0] Merge complete. Total scores: {len(score_list)}", flush=True)

        print(f"[rank 0] Canonicalizing scores...", flush=True)
        fname_list, score_list = canonicalize_scores_by_trial_records(
            trial_records=trial_records,
            utterance_ids=fname_list,
            scores=score_list,
            allow_duplicate_utterances=True,
        )
        print(f"[rank 0] Canonicalization done. {len(score_list)} scores.", flush=True)

    print(f"[rank 0] Building text list...", flush=True)
    text_list = [
        f"{r['speaker_id']} {r['utterance_id']} {s} {r['label_name']}"
        for r, fn, s in zip(trial_records, fname_list, score_list)
    ]
    print(f"[rank 0] Writing scores to {save_path}...", flush=True)
    with open(save_path, "w") as fh:
        fh.write("\n".join(text_list) + '\n')
    del text_list
    print(f"[rank 0] Scores file written.", flush=True)


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    import torch.nn as nn
    from torch.amp import autocast, GradScaler

    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()
    scaler = GradScaler('cuda')

    # Loss function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    defense_kwargs = get_defense_kwargs(config)
    defense_samples = get_defense_samples(config)
    use_non_blocking = device.type == "cuda"

    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        batch_x = batch_x.to(device, non_blocking=use_non_blocking)
        batch_y = batch_y.view(-1).type(torch.int64).to(
            device,
            non_blocking=use_non_blocking,
        )

        with autocast('cuda'):
            batch_out, _ = forward_with_defense(
                model=model,
                wav=batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
            )
            batch_loss = criterion(batch_out, batch_y)

        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optim)
        scaler.update()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError(f"scheduler error, got:{scheduler}")

    running_loss /= num_total
    return running_loss

if __name__ == "__main__":
    main(parse_args())
