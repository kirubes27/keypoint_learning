"""Build and hash the fixed cyclic RGB correspondence field."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch

try:
    from .material_transport_free_logits import (
        MaterialTransportConfig,
        build_bidirectional_field,
        extract_endpoint_grid_descriptors,
        local_candidate_layout,
    )
    from .material_transport_gate_io import (
        EXPECTED_FRAMES,
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        validate_sanitized_manifest,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_free_logits import (
        MaterialTransportConfig,
        build_bidirectional_field,
        extract_endpoint_grid_descriptors,
        local_candidate_layout,
    )
    from material_transport_gate_io import (
        EXPECTED_FRAMES,
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        validate_sanitized_manifest,
        write_json,
    )


FIELD_ARRAYS: dict[str, tuple[str, Any]] = {
    "forward_similarity": ("forward_similarity.npy", np.float32),
    "reverse_similarity": ("reverse_similarity.npy", np.float32),
    "forward_probability": ("forward_probability.npy", np.float32),
    "reverse_probability": ("reverse_probability.npy", np.float32),
    "forward_row_valid": ("forward_row_valid.npy", np.bool_),
    "reverse_row_valid": ("reverse_row_valid.npy", np.bool_),
    "forward_site_cost": ("forward_site_cost.npy", np.float32),
    "reverse_site_cost": ("reverse_site_cost.npy", np.float32),
    "forward_ambiguity": ("forward_ambiguity.npy", np.float32),
    "reverse_ambiguity": ("reverse_ambiguity.npy", np.float32),
    "forward_reciprocal_cost": ("forward_reciprocal_cost.npy", np.float32),
    "reverse_reciprocal_cost": ("reverse_reciprocal_cost.npy", np.float32),
    "forward_expected_displacement_cells": ("forward_expected_displacement_cells.npy", np.float32),
    "reverse_expected_displacement_cells": ("reverse_expected_displacement_cells.npy", np.float32),
    "forward_inactivity": ("forward_inactivity.npy", np.float32),
    "reverse_inactivity": ("reverse_inactivity.npy", np.float32),
}


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(name)
    if selected.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    if selected.type == "mps":
        require(torch.backends.mps.is_available(), "MPS requested but unavailable")
    return selected


def _load_rgb(path: Path, *, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / np.float32(255.0)
    require(array.shape == (512, 512, 3), f"RGB shape differs: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device=device)


class _DescriptorCache:
    def __init__(
        self,
        paths: list[Path],
        config: MaterialTransportConfig,
        device: torch.device,
        *,
        maximum_entries: int = 3,
    ) -> None:
        self.paths = paths
        self.config = config
        self.device = device
        self.maximum_entries = maximum_entries
        self.values: OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()

    def get(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frame_index in self.values:
            value = self.values.pop(frame_index)
            self.values[frame_index] = value
            return value
        image = _load_rgb(self.paths[frame_index], device=self.device)
        value = extract_endpoint_grid_descriptors(image, self.config)
        self.values[frame_index] = value
        while len(self.values) > self.maximum_entries:
            self.values.popitem(last=False)
        return value


def _allocate_arrays(
    output_dir: Path,
    *,
    edge_count: int,
    cells: int,
    candidates: int,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    matrix_names = {
        "forward_similarity",
        "reverse_similarity",
        "forward_probability",
        "reverse_probability",
    }
    for name, (filename, dtype) in FIELD_ARRAYS.items():
        shape = (edge_count, cells, candidates) if name in matrix_names else (edge_count, cells)
        arrays[name] = np.lib.format.open_memmap(
            output_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
    return arrays


def _field_numpy(field: dict[str, torch.Tensor | bool], name: str) -> np.ndarray:
    value = field[name]
    require(isinstance(value, torch.Tensor), f"field value {name} is not a tensor")
    return value.detach().cpu().numpy()


def _process_edges(
    edge_order: Iterable[int],
    *,
    paths: list[Path],
    config: MaterialTransportConfig,
    device: torch.device,
    arrays: dict[str, np.memmap],
    edge_row: dict[int, int],
    write: bool,
    verify_column_order: bool,
) -> tuple[bool, bool]:
    cache = _DescriptorCache(paths, config, device)
    frame_order_exact = True
    column_order_exact = True
    with torch.inference_mode():
        for frame_index in edge_order:
            target_frame = (int(frame_index) + 1) % EXPECTED_FRAMES
            source_descriptor, source_valid, _ = cache.get(int(frame_index))
            target_descriptor, target_valid, _ = cache.get(target_frame)
            field = build_bidirectional_field(
                source_descriptor,
                target_descriptor,
                source_valid,
                target_valid,
                config,
                verify_column_order=verify_column_order,
            )
            column_order_exact = column_order_exact and bool(field["column_order_reversal_exact"])
            row = edge_row[int(frame_index)]
            for name in FIELD_ARRAYS:
                value = _field_numpy(field, name)
                if write:
                    arrays[name][row] = value
                else:
                    frame_order_exact = frame_order_exact and bool(np.array_equal(arrays[name][row], value))
                    if not frame_order_exact:
                        raise RuntimeError(f"reverse frame processing changed {name} at edge {frame_index}")
    return frame_order_exact, column_order_exact


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(args.repo_root.is_dir(), "repository root missing")
    require(_git(args.repo_root, "status", "--porcelain") == "", "implementation worktree must be clean")
    manifest = load_json(args.manifest)
    validate_sanitized_manifest(manifest)
    require(_git(args.repo_root, "rev-parse", "HEAD") == manifest["implementation_head"], "implementation HEAD differs")
    for relative, expected in manifest["implementation_sources"].items():
        observed = file_record(args.repo_root / relative, include_path=False)
        require(observed == expected, f"implementation source differs: {relative}")
    config = MaterialTransportConfig(**manifest["field_config"])
    config.validate()
    paths = resolve_rgb_paths(manifest, object_root_override=args.data_object_root)
    edge_count = EXPECTED_FRAMES if args.edge_limit is None else int(args.edge_limit)
    require(1 <= edge_count <= EXPECTED_FRAMES, "edge_limit is outside 1..180")
    edges = list(range(edge_count))
    edge_row = {frame: row for row, frame in enumerate(edges)}
    selected_device = _device(args.device)
    if selected_device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True)
    candidate_index, candidate_valid, offsets_xy, self_column = local_candidate_layout(config)
    cells, candidates = candidate_index.shape
    arrays = _allocate_arrays(
        args.output_dir,
        edge_count=edge_count,
        cells=cells,
        candidates=candidates,
    )
    np.savez_compressed(
        args.output_dir / "FIELD_LAYOUT.npz",
        edge_frame_index=np.asarray(edges, dtype=np.int64),
        edge_target_frame_index=(np.asarray(edges, dtype=np.int64) + 1) % EXPECTED_FRAMES,
        candidate_index=candidate_index.cpu().numpy(),
        candidate_valid=candidate_valid.cpu().numpy(),
        offsets_xy=offsets_xy.cpu().numpy(),
        self_column=np.asarray(self_column, dtype=np.int64),
    )
    first_frame_exact, column_order_exact = _process_edges(
        edges,
        paths=paths,
        config=config,
        device=selected_device,
        arrays=arrays,
        edge_row=edge_row,
        write=True,
        verify_column_order=True,
    )
    require(first_frame_exact, "unexpected first-pass order status")
    for array in arrays.values():
        array.flush()
    reverse_frame_exact, replay_column_exact = _process_edges(
        reversed(edges),
        paths=paths,
        config=config,
        device=selected_device,
        arrays=arrays,
        edge_row=edge_row,
        write=False,
        verify_column_order=False,
    )
    require(reverse_frame_exact, "frame-order reversal changed a stored field")
    require(column_order_exact and replay_column_exact, "candidate-column order control failed")
    for array in arrays.values():
        array.flush()

    array_records = {
        name: file_record(args.output_dir / filename)
        for name, (filename, _) in FIELD_ARRAYS.items()
    }
    layout_record = file_record(args.output_dir / "FIELD_LAYOUT.npz")
    receipt = {
        "schema_version": "cyclic_rgb_sparse_field_receipt.v1",
        "artifact_type": "fixed_rgb_sparse_transition_field",
        "execution_scope": "complete" if edge_count == EXPECTED_FRAMES else "smoke",
        "edge_count": edge_count,
        "frame_count_bound": EXPECTED_FRAMES,
        "cell_count": cells,
        "candidate_count_per_row": candidates,
        "config": config.as_dict(),
        "manifest": file_record(args.manifest),
        "implementation_head": manifest["implementation_head"],
        "implementation_sources": manifest["implementation_sources"],
        "field_layout": layout_record,
        "field_arrays": array_records,
        "device": {
            "requested": args.device,
            "resolved": str(selected_device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "controls": {
            "candidate_column_reversal_exact_every_edge": bool(column_order_exact),
            "frame_processing_reversal_exact_every_array": bool(reverse_frame_exact),
            "all_rgb_hashes_rechecked_before_open": True,
            "invalid_rows_use_mass_preserving_self_loop": True,
            "privileged_evaluation_files_opened": False,
            "model_checkpoint_optimizer_or_geometry_used": False,
        },
        "runtime_seconds": float(time.time() - started),
        "statistical_scope": "fixed evidence construction; no statistical inference",
    }
    receipt_path = args.output_dir / "RGB_FIELD_RECEIPT.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-object-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--edge-limit", type=int)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
