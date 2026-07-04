"""Assert-based smoke gate for diagnostics/dxutils.py."""

from __future__ import annotations

import json

import numpy as np

from dxutils import (
    CELL64_NORM,
    derotate,
    estimate_rotation_model,
    highpass_residual,
    load_masks,
    load_run,
    mask_iou,
    run_directories,
    to_norm,
    to_px,
    trajectories,
    transport,
    warp_masks,
    write_rotation_report,
)


def main() -> None:
    masks = load_masks()
    assert masks.shape == (180, 512, 512)
    print("PASS masks: 180 binary frame-specific masks")

    rotation = estimate_rotation_model(masks)
    write_rotation_report(rotation)
    print(json.dumps(rotation.to_dict(), indent=2))
    assert rotation.geometry_ok, "G0 failed: stop before interpreting equivariance"
    print("PASS G0: mask transport IoU and split-centre stability")

    points = np.array([[-0.4, -0.2], [0.0, 0.0], [0.5, 0.3]], dtype=np.float64)
    recovered = transport(transport(points, 7, rotation), -7, rotation)
    assert np.max(np.abs(recovered - points)) < 1e-10
    assert np.max(np.abs(to_norm(to_px(points)) - points)) < 1e-10
    print("PASS coordinate and transport inverses")

    warped_180 = warp_masks(masks[[0]], rotation.sign * 180.0, rotation.center_px)
    iou_180 = float(mask_iou(warped_180, masks[[90]])[0])
    assert iou_180 >= 0.93, f"180-degree mask transport IoU too low: {iou_180}"
    print(f"PASS 180-degree output check: IoU={iou_180:.4f}")

    runs = run_directories()
    for name, run_dir in runs.items():
        extractor, cfg = load_run(run_dir)
        coords = trajectories(extractor)
        assert coords.shape == (180, 10, 2)
        assert np.isfinite(coords).all()
        expected_res = 128 if name == "smoke128" else 64
        assert extractor.heatmap_res == expected_res
        print(f"PASS {name}: strict checkpoint load, trajectory {coords.shape}")
        if name == "task80":
            residual = highpass_residual(derotate(coords, rotation))
            channel_sigma = residual.std(axis=0).mean(axis=1) / CELL64_NORM
            median_sigma = float(np.median(channel_sigma))
            assert 0.80 <= median_sigma <= 1.10, median_sigma
            print(f"PASS frozen jitter reproduction: median={median_sigma:.3f} cells64")

    print("\nALL DXUTILS TESTS PASSED")


if __name__ == "__main__":
    main()
