import os
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
from typing import Iterator, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class DistributedEvalRuntime:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


class DeterministicShardSampler(Sampler[int]):
    def __init__(self, data_source: Sequence[T_co], num_replicas: int, rank: int) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be at least 1, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must satisfy 0 <= rank < num_replicas, got rank={rank}, "
                f"num_replicas={num_replicas}."
            )
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        dataset_length = len(self.data_source)
        shard_length = dataset_length // self.num_replicas
        remainder = dataset_length % self.num_replicas
        start = self.rank * shard_length + min(self.rank, remainder)
        stop = start + shard_length + (1 if self.rank < remainder else 0)
        return iter(range(start, stop))

    def __len__(self) -> int:
        dataset_length = len(self.data_source)
        shard_length = dataset_length // self.num_replicas
        if self.rank < dataset_length % self.num_replicas:
            return shard_length + 1
        return shard_length


def distributed_launch_requested() -> bool:
    return any(
        os.environ.get(name) is not None for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    )


def _parse_distributed_env_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got '{value}'."
        ) from exc


def _resolve_process_group_timeout() -> timedelta:
    timeout_value = os.environ.get("CS199_DISTRIBUTED_TIMEOUT_SECONDS")
    if timeout_value is None:
        # PGD eval can leave faster ranks waiting well beyond PyTorch's
        # default 10-minute timeout before the first cross-rank sync.
        return timedelta(hours=1)

    timeout_seconds = _parse_distributed_env_int(
        "CS199_DISTRIBUTED_TIMEOUT_SECONDS",
        timeout_value,
    )
    if timeout_seconds < 1:
        raise ValueError(
            "CS199_DISTRIBUTED_TIMEOUT_SECONDS must be at least 1 second, "
            f"got {timeout_seconds}."
        )
    return timedelta(seconds=timeout_seconds)


def initialize_distributed_eval_runtime() -> DistributedEvalRuntime:
    env_values = {
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "RANK": os.environ.get("RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }
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
    dist.init_process_group(
        backend="nccl",
        timeout=_resolve_process_group_timeout(),
    )
    return DistributedEvalRuntime(
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed_eval_runtime(runtime: Optional[DistributedEvalRuntime]) -> None:
    if runtime is not None and runtime.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def uses_multi_rank_distribution(runtime: Optional[DistributedEvalRuntime]) -> bool:
    return runtime is not None and runtime.is_distributed and runtime.world_size > 1


def is_primary_rank(runtime: Optional[DistributedEvalRuntime]) -> bool:
    return runtime is None or runtime.rank == 0


def distributed_barrier(runtime: Optional[DistributedEvalRuntime]) -> None:
    if runtime is not None and runtime.is_distributed:
        dist.barrier()


def broadcast_from_primary(runtime: Optional[DistributedEvalRuntime], payload):
    if runtime is None or not runtime.is_distributed:
        return payload
    payload_container = [payload if runtime.rank == 0 else None]
    dist.broadcast_object_list(payload_container, src=0)
    return payload_container[0]


def rank_temp_path(run_dir: Path, artifact_stem: str, runtime: DistributedEvalRuntime) -> Path:
    temp_root = run_dir / ".dist_eval"
    return temp_root / f"{artifact_stem}.rank{runtime.rank:05d}.json"


def rank_temp_dir(final_dir: Path, runtime: DistributedEvalRuntime) -> Path:
    temp_root = final_dir.parent / ".dist_eval"
    return temp_root / f"{final_dir.name}.rank{runtime.rank:05d}"
