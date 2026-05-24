import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pgd_utils import DEFAULT_CLAMP_MAX, DEFAULT_CLAMP_MIN, DEFAULT_PGD_STEPS


DEFAULT_PGD_EPSILON = 0.001


def _parse_gpu_ids(raw_value: Optional[str]) -> Optional[List[int]]:
    if raw_value is None:
        return None

    gpu_ids: List[int] = []
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            gpu_id = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid GPU id '{stripped}'. Expected a comma-separated list of integers."
            ) from exc
        if gpu_id < 0:
            raise argparse.ArgumentTypeError(
                f"GPU ids must be non-negative, got {gpu_id}."
            )
        gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise argparse.ArgumentTypeError(
            "Expected at least one GPU id when --gpu-ids is provided."
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError(
            f"GPU ids must be unique, got {raw_value!r}."
        )
    return gpu_ids


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only entry point for matched clean and PGD ASVspoof5 scoring."
    )
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the pretrained anti-spoofing checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory where score and metric files will be written.",
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=["dev", "eval"],
        help="Dataset split to score.",
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
        "--trial-file",
        "--trial_file",
        dest="trial_file",
        default=None,
        help="Optional explicit dev/eval trial file override. Keep it aligned with --split.",
    )
    parser.add_argument(
        "--audio-root",
        "--audio_root",
        dest="audio_root",
        default=None,
        help="Optional explicit audio directory override for the selected split.",
    )
    parser.add_argument(
        "--ssl-pretrained-path",
        "--ssl_pretrained_path",
        dest="ssl_pretrained_path",
        default=None,
        help="Optional WavLM checkpoint override.",
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
        "--epsilon",
        type=float,
        default=DEFAULT_PGD_EPSILON,
        help="PGD epsilon. Defaults to 0.001 for the practical PGD-5 baseline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_PGD_STEPS,
        help="PGD step count. Defaults to the practical PGD-5 baseline.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="PGD step size. If omitted, the script uses epsilon / steps.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Enable random start inside the Linf epsilon ball before iterative updates.",
    )
    parser.add_argument(
        "--clean-score-filename",
        "--clean_score_filename",
        dest="clean_score_filename",
        default="clean_scores.txt",
        help="Filename for clean scores in the matched clean-vs-PGD run.",
    )
    parser.add_argument(
        "--adv-score-filename",
        "--adv_score_filename",
        dest="adv_score_filename",
        default=None,
        help="Optional custom adversarial score filename.",
    )
    parser.add_argument(
        "--clamp-min",
        "--clamp_min",
        dest="clamp_min",
        type=float,
        default=DEFAULT_CLAMP_MIN,
        help="Minimum clamp value for adversarial waveforms.",
    )
    parser.add_argument(
        "--clamp-max",
        "--clamp_max",
        dest="clamp_max",
        type=float,
        default=DEFAULT_CLAMP_MAX,
        help="Maximum clamp value for adversarial waveforms.",
    )
    parser.add_argument(
        "--save-adv-audio",
        action="store_true",
        help="Optionally save PGD-perturbed waveforms as .wav files under the run output directory.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute and print metrics without requiring score, metric, or summary artifact writes.",
    )
    parser.add_argument(
        "--skip-clean-pass",
        action="store_true",
        help="Run PGD scoring without a clean pass. Clean comparison fields are omitted.",
    )
    parser.add_argument(
        "--num-gpus",
        "--num_gpus",
        dest="num_gpus",
        type=int,
        default=1,
        help=(
            "Number of GPUs to use for PGD evaluation. "
            "When greater than 1, the script spawns one worker per GPU and keeps the "
            "parent process as a CPU-only coordinator."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        "--gpu_ids",
        dest="gpu_ids",
        type=_parse_gpu_ids,
        default=None,
        help=(
            "Optional comma-separated CUDA device ids to use in multi-GPU mode, "
            'for example "0,1". Defaults to the first --num-gpus visible devices.'
        ),
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=0,
        help="DataLoader worker count to use inside each scoring worker process.",
    )
    return parser.parse_args(argv)


def _print_runtime_notes(args: argparse.Namespace) -> None:
    resolved_alpha = args.alpha
    if resolved_alpha is None:
        resolved_alpha = 0.0 if args.steps == 0 else args.epsilon / args.steps

    print("Attack: untargeted Linf PGD")
    print(
        "Attack settings: "
        f"epsilon={args.epsilon:.6f}, "
        f"steps={args.steps}, "
        f"alpha={resolved_alpha:.6f}, "
        f"random_start={args.random_start}, "
        f"clamp=[{args.clamp_min:.6f}, {args.clamp_max:.6f}]"
    )
    print(f"Output root: {args.output_dir}")
    print(
        "Execution: "
        f"num_gpus={args.num_gpus}, "
        f"gpu_ids={args.gpu_ids if args.gpu_ids is not None else 'auto'}, "
        f"num_workers={args.num_workers}"
    )

    if args.steps > DEFAULT_PGD_STEPS:
        print(
            "Runtime note: "
            f"steps={args.steps} is more expensive than the default PGD-{DEFAULT_PGD_STEPS} run."
        )
    if args.save_adv_audio:
        print(
            "Runtime note: "
            "saving adversarial waveforms increases I/O, disk usage, and total runtime."
        )
    if args.random_start:
        print(
            "Runtime note: "
            "random_start=True reduces exact reproducibility across repeated runs."
        )
    if args.metrics_only:
        print("Runtime note: metrics_only=True disables score, metric, and summary file writes.")
    if args.skip_clean_pass:
        print("Runtime note: skip_clean_pass=True disables clean scoring and clean-vs-PGD deltas.")
    if args.num_gpus > 1:
        print(
            "Runtime note: multi-GPU mode keeps the parent process off the attack path "
            "and uses one dedicated PGD worker per GPU."
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    from src.eval_utils import run_pgd_scoring_pipeline

    args = parse_args(argv)
    _print_runtime_notes(args)

    result = run_pgd_scoring_pipeline(
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output_dir,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_root=args.metadata_root,
        trial_file=args.trial_file,
        audio_root=args.audio_root,
        ssl_pretrained_path=args.ssl_pretrained_path,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        steps=args.steps,
        alpha=args.alpha,
        random_start=args.random_start,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        clean_score_filename=args.clean_score_filename,
        adv_score_filename=args.adv_score_filename,
        save_adv_audio=args.save_adv_audio,
        metrics_only=args.metrics_only,
        skip_clean_pass=args.skip_clean_pass,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        num_workers=args.num_workers,
    )

    print(f"Scored split: {result['split']}")
    print(f"Checkpoint: {result['weights_path']}")
    print(f"Trial file: {result['trial_path']}")
    if result["clean_score_path"] is not None:
        print(f"Clean scores: {result['clean_score_path']}")
    if result["adv_score_path"] is not None:
        print(f"Adversarial scores: {result['adv_score_path']}")
    if result["clean_metrics_path"] is not None:
        print(f"Clean metrics: {result['clean_metrics_path']}")
    if result["adv_metrics_path"] is not None:
        print(f"Adversarial metrics: {result['adv_metrics_path']}")
    if result["adv_audio_dir"] is not None:
        print(f"Adversarial audio: {result['adv_audio_dir']}")
    if result["summary_json_path"] is not None:
        print(f"Summary JSON: {result['summary_json_path']}")
    if result["summary_text_path"] is not None:
        print(f"Summary TXT: {result['summary_text_path']}")
    print(f"Utterances scored: {result['utterance_count']}")
    if not result["skip_clean_pass"]:
        print(
            "Clean metrics: "
            f"minDCF={result['metrics']['min_dcf']['clean']:.6f}, "
            f"EER={result['metrics']['eer']['clean'] * 100:.6f}%, "
            f"CLLR={result['metrics']['cllr']['clean'] * 100:.6f}%"
        )
    print(
        "Adversarial metrics: "
        f"minDCF={result['metrics']['min_dcf']['adversarial']:.6f}, "
        f"EER={result['metrics']['eer']['adversarial'] * 100:.6f}%, "
        f"CLLR={result['metrics']['cllr']['adversarial'] * 100:.6f}%"
    )
    if result["attack_stats"]:
        print(
            "PGD stats: "
            f"epsilon={result['attack_stats']['epsilon']:.6f}, "
            f"steps={result['attack_stats']['steps']:.0f}, "
            f"alpha={result['attack_stats']['alpha']:.6f}, "
            f"loss={result['attack_stats']['loss']:.6f}, "
            f"max_abs={result['attack_stats']['max_abs_perturbation']:.6f}, "
            f"mean_abs={result['attack_stats']['mean_abs_perturbation']:.6f}, "
            f"mean_l2={result['attack_stats']['mean_l2_perturbation']:.6f}"
        )
    for warning in result["artifact_warnings"]:
        print(f"Artifact warning: {warning}")


if __name__ == "__main__":
    main()
