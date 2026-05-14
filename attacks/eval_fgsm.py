import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only entry point for clean and FGSM ASVspoof5 scoring."
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
        "--defense-config",
        "--defense_config",
        dest="defense_config",
        default=None,
        help="Optional shared defense config JSON override.",
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
        "--score-filename",
        "--score_filename",
        dest="score_filename",
        default=None,
        help="Optional custom score filename.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="FGSM epsilon. If provided, the script writes matched clean and adversarial score files.",
    )
    parser.add_argument(
        "--clean-score-filename",
        "--clean_score_filename",
        dest="clean_score_filename",
        default="clean_scores.txt",
        help="Filename for clean scores when running FGSM scoring.",
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
        default=-1.0,
        help="Minimum clamp value for adversarial waveforms.",
    )
    parser.add_argument(
        "--clamp-max",
        "--clamp_max",
        dest="clamp_max",
        type=float,
        default=1.0,
        help="Maximum clamp value for adversarial waveforms.",
    )
    parser.add_argument(
        "--save-adv-audio",
        action="store_true",
        help="Optionally save FGSM-perturbed waveforms as .wav files under the run output directory.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute and print metrics without requiring score, metric, or summary artifact writes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible stochastic defense evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    from src.eval_utils import run_clean_evaluation, run_fgsm_scoring_pipeline

    args = parse_args()
    if args.epsilon is not None:
        result = run_fgsm_scoring_pipeline(
            config_path=args.config,
            weights_path=args.weights,
            output_dir=args.output_dir,
            split=args.split,
            dataset_root=args.dataset_root,
            metadata_root=args.metadata_root,
            trial_file=args.trial_file,
            audio_root=args.audio_root,
            ssl_pretrained_path=args.ssl_pretrained_path,
            defense_config_path=args.defense_config,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            clean_score_filename=args.clean_score_filename,
            adv_score_filename=args.adv_score_filename,
            save_adv_audio=args.save_adv_audio,
            metrics_only=args.metrics_only,
            seed=args.seed,
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
                "FGSM stats: "
                f"epsilon={result['attack_stats']['epsilon']:.6f}, "
                f"loss={result['attack_stats']['loss']:.6f}, "
                f"max_abs={result['attack_stats']['max_abs_perturbation']:.6f}, "
                f"mean_abs={result['attack_stats']['mean_abs_perturbation']:.6f}, "
                f"mean_l2={result['attack_stats']['mean_l2_perturbation']:.6f}"
            )
        for warning in result["artifact_warnings"]:
            print(f"Artifact warning: {warning}")
        return

    result = run_clean_evaluation(
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output_dir,
        split=args.split,
        dataset_root=args.dataset_root,
        metadata_root=args.metadata_root,
        trial_file=args.trial_file,
        audio_root=args.audio_root,
        ssl_pretrained_path=args.ssl_pretrained_path,
        defense_config_path=args.defense_config,
        batch_size=args.batch_size,
        score_filename=args.score_filename,
        metrics_only=args.metrics_only,
        seed=args.seed,
    )
    print(f"Scored split: {result['split']}")
    print(f"Checkpoint: {result['weights_path']}")
    if result["score_path"] is not None:
        print(f"Score file: {result['score_path']}")
    if result["metrics_path"] is not None:
        print(f"Metric file: {result['metrics_path']}")
    print(
        "Metrics: "
        f"minDCF={result['min_dcf']:.6f}, "
        f"EER={result['eer'] * 100:.6f}%, "
        f"CLLR={result['cllr'] * 100:.6f}%"
    )
    for warning in result["artifact_warnings"]:
        print(f"Artifact warning: {warning}")


if __name__ == "__main__":
    main()
