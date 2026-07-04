#!/usr/bin/env python3
"""Write one sweep --filter expression per full-factorial hammer config."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path


SWEEP_GRID = {
    "lambda_act": [0.0, 0.5, 1.0],
    "lambda_cycle": [0.0, 0.1],
    "lambda_disp": [0.0, 0.1],
    "lambda_ent": [0.0, 0.01, 0.05],
    "lambda_inv": [0.0, 0.1],
    "lambda_loc": [0.0, 0.01],
    "lambda_smooth": [0.0, 0.001, 0.01],
}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cluster/hammer_sweep_filters.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(SWEEP_GRID)
    lines = []
    for values in itertools.product(*(SWEEP_GRID[k] for k in keys)):
        lines.append(",".join(f"{k}={v:g}" for k, v in zip(keys, values)))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} filters to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
