from __future__ import annotations

import unittest

import torch

import numpy as np

from keypoint_net.evaluate_material_transport_witness_distribution_replay import (
    CELL_STEP_PX,
    GRID_SIZE,
    bilinear_grid_distribution,
    hard_centered_local_readout,
    transport_distribution,
)
from keypoint_net.evaluate_material_transport_local_motion_replay import (
    condition_probability,
    radius_column_mask,
)

from keypoint_net.material_transport_free_logits import (
    MaterialTransportConfig,
    build_bidirectional_field,
    channel_overlap_loss,
    conditional_from_similarity,
    extract_endpoint_grid_descriptors,
    jensen_shannon_divergence,
    local_candidate_layout,
    sparse_transport,
    weighted_site_loss,
)


class MaterialTransportFreeLogitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MaterialTransportConfig(
            image_size=64,
            grid_size=8,
            patch_size=9,
            search_radius_cells=2,
            correspondence_temperature=0.02,
            minimum_patch_rms=1.0e-6,
            minimum_motion_cells=0.125,
            invalid_site_cost=2.0,
            descriptor_chunk_size=16,
        )

    def test_planted_rgb_translation_is_bidirectional_and_order_exact(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        source = torch.rand((3, 64, 64), generator=generator)
        target = torch.zeros_like(source)
        target[:, :, 9:] = source[:, :, :-9]
        source_descriptor, source_valid, _ = extract_endpoint_grid_descriptors(
            source, self.config
        )
        target_descriptor, target_valid, _ = extract_endpoint_grid_descriptors(
            target, self.config
        )
        field = build_bidirectional_field(
            source_descriptor,
            target_descriptor,
            source_valid,
            target_valid,
            self.config,
            verify_column_order=True,
        )
        candidate_index = field["candidate_index"]
        offsets = field["offsets_xy"]
        source_cell = 3 * 8 + 3
        target_cell = 3 * 8 + 4
        forward_column = int(torch.argmax(field["forward_probability"][source_cell]).item())
        reverse_column = int(torch.argmax(field["reverse_probability"][target_cell]).item())
        self.assertEqual(candidate_index[source_cell, forward_column].item(), target_cell)
        self.assertEqual(candidate_index[target_cell, reverse_column].item(), source_cell)
        self.assertEqual(offsets[forward_column].tolist(), [1, 0])
        self.assertEqual(offsets[reverse_column].tolist(), [-1, 0])
        self.assertTrue(field["column_order_reversal_exact"])

    def test_duplicate_target_has_more_ambiguity_than_unique_target(self) -> None:
        config = MaterialTransportConfig(
            image_size=32,
            grid_size=5,
            patch_size=5,
            search_radius_cells=1,
            correspondence_temperature=0.05,
            descriptor_chunk_size=8,
        )
        cells = 25
        source = torch.eye(cells)
        target_unique = torch.roll(torch.eye(cells), shifts=1, dims=0)
        valid = torch.ones(cells, dtype=torch.bool)
        unique = build_bidirectional_field(source, target_unique, valid, valid, config)

        target_duplicate = target_unique.clone()
        row = 2 * 5 + 2
        right = row + 1
        left = row - 1
        target_duplicate[left] = source[row]
        target_duplicate[right] = source[row]
        duplicate = build_bidirectional_field(source, target_duplicate, valid, valid, config)
        self.assertGreater(
            duplicate["forward_ambiguity"][row].item(),
            unique["forward_ambiguity"][row].item() + 0.1,
        )

    def test_static_distinctive_site_costs_more_than_moving_site(self) -> None:
        config = MaterialTransportConfig(
            image_size=32,
            grid_size=5,
            patch_size=5,
            search_radius_cells=1,
            correspondence_temperature=0.01,
            minimum_motion_cells=0.125,
            descriptor_chunk_size=8,
        )
        cells = 25
        source = torch.eye(cells)
        target = torch.eye(cells)
        moving_source = 2 * 5 + 2
        moving_target = moving_source + 1
        displaced_identity = source[moving_source].clone()
        target[moving_source] = torch.zeros(cells)
        target[moving_source, (moving_source + 7) % cells] = 1.0
        target[moving_target] = displaced_identity
        valid = torch.ones(cells, dtype=torch.bool)
        field = build_bidirectional_field(source, target, valid, valid, config)
        static_source = 1 * 5 + 1
        self.assertLess(
            field["forward_site_cost"][moving_source].item(),
            field["forward_site_cost"][static_source].item(),
        )
        self.assertGreater(field["forward_inactivity"][static_source].item(), 0.99)
        self.assertLess(field["forward_inactivity"][moving_source].item(), 1.0e-5)

    def test_invalid_row_is_mass_preserving_self_loop_with_cost_two(self) -> None:
        candidate_index, candidate_valid, _, self_column = local_candidate_layout(self.config)
        similarity = torch.full(candidate_index.shape, -torch.inf)
        probability, row_valid, _ = conditional_from_similarity(
            similarity, candidate_valid, self_column, self.config
        )
        self.assertFalse(bool(row_valid.any()))
        self.assertTrue(torch.all(probability[:, self_column] == 1.0))
        self.assertTrue(torch.all(probability.sum(dim=1) == 1.0))

        descriptor = torch.eye(64)
        valid = torch.zeros(64, dtype=torch.bool)
        field = build_bidirectional_field(descriptor, descriptor, valid, valid, self.config)
        self.assertTrue(
            torch.all(field["forward_site_cost"] == self.config.invalid_site_cost)
        )

    def test_correct_transport_has_lower_js_than_wrong_site(self) -> None:
        config = MaterialTransportConfig(
            image_size=32,
            grid_size=5,
            patch_size=5,
            search_radius_cells=1,
            correspondence_temperature=0.01,
            descriptor_chunk_size=8,
        )
        cells = 25
        source_descriptor = torch.eye(cells)
        target_descriptor = torch.roll(torch.eye(cells), shifts=1, dims=0)
        valid = torch.ones(cells, dtype=torch.bool)
        field = build_bidirectional_field(
            source_descriptor, target_descriptor, valid, valid, config
        )
        source_probability = torch.zeros((1, 1, cells))
        source_probability[0, 0, 12] = 1.0
        prediction = sparse_transport(
            source_probability,
            field["forward_probability"].unsqueeze(0),
            field["candidate_index"],
        )
        correct = prediction.clone()
        wrong = torch.zeros_like(prediction)
        wrong[0, 0, 0] = 1.0
        self.assertLess(
            jensen_shannon_divergence(prediction, correct).item(),
            jensen_shannon_divergence(prediction, wrong).item(),
        )

    def test_duplicate_channels_have_larger_overlap(self) -> None:
        duplicate = torch.zeros((1, 2, 16))
        duplicate[:, :, 5] = 1.0
        separated = torch.zeros_like(duplicate)
        separated[:, 0, 5] = 1.0
        separated[:, 1, 9] = 1.0
        self.assertEqual(channel_overlap_loss(duplicate).item(), 1.0)
        self.assertEqual(channel_overlap_loss(separated).item(), 0.0)

    def test_location_dependent_site_cost_has_gradient_at_uniform_logits(self) -> None:
        logits = torch.zeros((2, 3, 16), requires_grad=True)
        probability = torch.softmax(logits, dim=2)
        site_cost = torch.stack(
            (torch.linspace(0.0, 1.0, 16), torch.linspace(1.0, 0.0, 16)), dim=0
        )
        loss = weighted_site_loss(probability, site_cost)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(torch.linalg.vector_norm(logits.grad).item(), 1.0e-5)

    def test_privileged_bilinear_replay_reproduces_subcell_coordinate(self) -> None:
        coordinate = np.asarray([213.25, 377.75], dtype=np.float64)
        probability = bilinear_grid_distribution(coordinate)
        self.assertEqual(np.count_nonzero(probability), 4)
        grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float64), GRID_SIZE)
        grid_y = np.repeat(np.arange(GRID_SIZE, dtype=np.float64), GRID_SIZE)
        reproduced = np.asarray(
            [probability @ grid_x, probability @ grid_y], dtype=np.float64
        ) * CELL_STEP_PX
        np.testing.assert_allclose(reproduced, coordinate, atol=1.0e-10)

    def test_distribution_replay_identity_field_preserves_local_readout(self) -> None:
        coordinate = np.asarray([213.25, 377.75], dtype=np.float64)
        source = bilinear_grid_distribution(coordinate)
        cells = GRID_SIZE * GRID_SIZE
        candidate_index = np.arange(cells, dtype=np.int64)[:, None]
        conditional = np.ones((cells, 1), dtype=np.float64)
        transported = transport_distribution(source, conditional, candidate_index)
        readout = hard_centered_local_readout(transported)
        np.testing.assert_allclose(readout["coordinate_px"], coordinate, atol=1.0e-10)

    def test_radius_two_conditioning_retains_exactly_twenty_five_columns(self) -> None:
        offsets = np.asarray(
            [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)],
            dtype=np.int64,
        )
        keep = radius_column_mask(offsets)
        self.assertEqual(int(np.sum(keep)), 25)
        self.assertTrue(bool(keep[40]))
        self.assertEqual(np.max(np.abs(offsets[keep])), 2)

    def test_radius_two_conditioning_is_candidate_order_invariant(self) -> None:
        offsets = np.asarray(
            [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)],
            dtype=np.int64,
        )
        keep = radius_column_mask(offsets)
        generator = np.random.default_rng(42)
        probability = generator.random((7, 81), dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        forward = condition_probability(probability, keep)
        reversed_result = condition_probability(probability[:, ::-1], keep[::-1])[:, ::-1]
        np.testing.assert_allclose(forward, reversed_result, atol=1.0e-15, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
