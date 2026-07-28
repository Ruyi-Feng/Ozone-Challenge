"""CLI: python -m model_baseline.scripts.run_train --config configs/model_baseline.yaml"""

from __future__ import annotations

import argparse

from model_baseline.train import main as train_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conflict baseline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/model_baseline.yaml",
        help="Path to model_baseline YAML config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_main(config_path=args.config)


if __name__ == "__main__":
    main()
