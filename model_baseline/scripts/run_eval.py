"""CLI: python -m model_baseline.scripts.run_eval --config configs/model_baseline.yaml"""

from __future__ import annotations

import argparse

from model_baseline.evaluate import main as eval_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate conflict baseline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/model_baseline.yaml",
        help="Path to model_baseline YAML config",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_main(config_path=args.config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
