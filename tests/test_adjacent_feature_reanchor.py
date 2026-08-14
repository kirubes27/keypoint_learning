from __future__ import annotations

import numpy as np
import pytest
import torch

from keypoint_net.adjacent_feature_reanchor import (
    AdjacentFeatureReanchorError,
    cyclic_target_indices,
    endpoint_coordinates_to_cells,
    paired_cosine_correlation_maps,
    sample_paired_descriptors,
    sample_paired_target_similarities,
)
from keypoint_net.frozen_feature_decode import endpoint_cells_to_coordinates, stable_spatial_top_two
from keypoint_net.evaluate_adjacent_feature_reanchor import MATERIAL_ERROR_MAX, adjacent_metrics


def test_cyclic_target_indices_include_seam() -> None:
    np.testing.assert_array_equal(cyclic_target_indices(4), [1, 2, 3, 0])


def test_endpoint_coordinate_roundtrip_uses_xy_cells() -> None:
    x = np.asarray([[0, 63], [7, 19]])
    y = np.asarray([[63, 0], [11, 23]])
    coordinates = endpoint_cells_to_coordinates(x, y, size=64)
    cells = endpoint_coordinates_to_cells(coordinates, size=64)
    np.testing.assert_array_equal(cells[..., 0], x)
    np.testing.assert_array_equal(cells[..., 1], y)


def test_off_grid_coordinate_is_rejected() -> None:
    with pytest.raises(AdjacentFeatureReanchorError, match="not on endpoint grid"):
        endpoint_coordinates_to_cells(np.asarray([[0.0, 0.0]]), size=64)


def test_paired_self_and_adjacent_decode_known_cells() -> None:
    fields = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    # Source frame 0: KP0 at (x=1,y=2), KP1 at (x=3,y=0).
    fields[0, :, 2, 1] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    fields[0, :, 0, 3] = torch.tensor([0.0, 3.0, 0.0, 0.0])
    # Source frame 1 / cyclic target for frame 0: identities move to known cells.
    fields[1, :, 1, 2] = torch.tensor([5.0, 0.0, 0.0, 0.0])
    fields[1, :, 3, 0] = torch.tensor([0.0, 7.0, 0.0, 0.0])
    source_xy = endpoint_cells_to_coordinates(np.asarray([[1, 3], [2, 0]]), np.asarray([[2, 0], [1, 3]]), size=4)
    source = torch.from_numpy(source_xy.astype(np.float32))
    _, raw_norm, descriptors = sample_paired_descriptors(fields, source)
    torch.testing.assert_close(raw_norm, torch.tensor([[2.0, 3.0], [5.0, 7.0]]))

    self_maps = paired_cosine_correlation_maps(descriptors, fields).numpy()
    self_top = stable_spatial_top_two(self_maps, exclusion_radius_cells=1.0)
    np.testing.assert_array_equal(self_top["top1_x_cell"], [[1, 3], [2, 0]])
    np.testing.assert_array_equal(self_top["top1_y_cell"], [[2, 0], [1, 3]])

    target = fields[torch.tensor([1, 0])]
    adjacent_maps = paired_cosine_correlation_maps(descriptors, target).numpy()
    adjacent_top = stable_spatial_top_two(adjacent_maps, exclusion_radius_cells=1.0)
    np.testing.assert_array_equal(adjacent_top["top1_x_cell"], [[2, 0], [1, 3]])
    np.testing.assert_array_equal(adjacent_top["top1_y_cell"], [[1, 3], [2, 0]])


def test_target_similarity_matches_paired_map_at_exact_cells() -> None:
    generator = torch.Generator().manual_seed(19)
    fields = torch.randn((3, 5, 4, 4), generator=generator)
    x = np.asarray([[0, 3], [1, 2], [3, 0]])
    y = np.asarray([[3, 0], [2, 1], [0, 3]])
    coords = torch.from_numpy(endpoint_cells_to_coordinates(x, y, size=4).astype(np.float32))
    _, _, descriptors = sample_paired_descriptors(fields, coords)
    maps = paired_cosine_correlation_maps(descriptors, fields)
    similarities = sample_paired_target_similarities(descriptors, fields, coords)
    for batch in range(3):
        for channel in range(2):
            torch.testing.assert_close(similarities[batch, channel], maps[batch, channel, y[batch, channel], x[batch, channel]])


def test_planted_adjacent_nearest_cells_pass_material_and_self_controls() -> None:
    source_x = np.tile(np.arange(10, dtype=np.int64) + 20, (180, 1))
    source_y = np.tile(np.arange(10, dtype=np.int64) + 25, (180, 1))
    source_x = np.remainder(source_x + (np.arange(180)[:, None] // 3), 40) + 12
    source_y = np.remainder(source_y + (np.arange(180)[:, None] // 5), 40) + 12
    source = endpoint_cells_to_coordinates(source_x, source_y, size=64)
    radians = np.deg2rad(2.0)
    physical = np.empty_like(source)
    physical[..., 0] = np.cos(radians) * source[..., 0] - np.sin(radians) * source[..., 1]
    physical[..., 1] = np.sin(radians) * source[..., 0] + np.cos(radians) * source[..., 1]
    target_cells = np.rint((physical + 1.0) * 31.5).astype(np.int64)
    adjacent = endpoint_cells_to_coordinates(target_cells[..., 0], target_cells[..., 1], size=64)
    masks = np.ones((180, 512, 512), dtype=bool)
    report, arrays = adjacent_metrics(
        source=source,
        target_detector=adjacent.copy(),
        self_decoded=source.copy(),
        adjacent_decoded=adjacent,
        raw_norm=np.ones((180, 10), dtype=np.float64),
        self_margin=np.ones((180, 10), dtype=np.float64),
        adjacent_margin=np.ones((180, 10), dtype=np.float64),
        masks=masks,
        target_index=np.remainder(np.arange(180) + 1, 180),
    )
    assert report["self_retrieval_same_cell_count"] == 1800
    assert report["adjacent_material_success_count"] == 1800
    assert float(np.max(arrays["feature_material_error_pixels"])) <= MATERIAL_ERROR_MAX * 255.5
