"""Run one frozen 64-versus-128 development cell.

The public interface intentionally exposes only the cell, corpus, and output
root.  Scientific settings are resolved from the committed experiment
manifest and passed to ``train.py`` as an exact internal argument set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_net import representation_fresh_checkpoint_authorization as fresh


def build_command(
    *,
    cell_id: str,
    data_root: str,
    output_root: str,
) -> list[str]:
    arguments = fresh.training_arguments(
        REPO_ROOT,
        cell_id,
        data_root=data_root,
        output_root=output_root,
    )
    return [
        sys.executable,
        str((REPO_ROOT / "keypoint_net/train.py").resolve()),
        *fresh.command_arguments(arguments),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one exact frozen roll head-package cell"
    )
    parser.add_argument("--cell_id", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--print_command",
        action="store_true",
        help="Print the resolved command without starting training.",
    )
    args = parser.parse_args(argv)
    command = build_command(
        cell_id=args.cell_id,
        data_root=args.data_root,
        output_root=args.output_root,
    )
    if args.print_command:
        print(" ".join(command))
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
