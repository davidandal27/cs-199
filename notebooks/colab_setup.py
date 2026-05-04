import argparse
import json
from pprint import pprint

from src.path_utils import resolve_workflow_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Colab and Google Drive paths.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where runtime artifacts should be written.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional anti-spoofing model checkpoint to validate.",
    )
    parser.add_argument(
        "--skip_training_assets",
        action="store_true",
        help="Skip MUSAN/RIR validation when only evaluation assets are needed.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    paths = resolve_workflow_paths(
        config=config,
        output_dir=args.output_dir,
        model_weights_path=args.weights,
        require_training_assets=not args.skip_training_assets,
    )

    pprint(paths)


if __name__ == "__main__":
    main()
