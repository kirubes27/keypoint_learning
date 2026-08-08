"""CPU semantic, gradient, sampling, and zero-arm tests for attachment loss."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


KEYPOINT_NET_ROOT = Path(__file__).resolve().parents[1] / "keypoint_net"
if str(KEYPOINT_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_NET_ROOT))

from keypoint_net import descriptor_attachment as attachment
from keypoint_net import model as model_module
from keypoint_net import train as train_module


def _exact_features(*, requires_grad: bool = False, constant: float | None = None):
    if constant is None:
        tensor = torch.randn(1, 128, 64, 64)
    else:
        tensor = torch.full((1, 128, 64, 64), constant)
    return tensor.requires_grad_(requires_grad)


def _coordinates(*, requires_grad: bool = False):
    coordinates = torch.linspace(-0.7, 0.7, 20).view(1, 10, 2)
    return coordinates.clone().requires_grad_(requires_grad)


class DescriptorSemanticsTests(unittest.TestCase):
    def test_one_hot_expected_ce_permutation_and_labels(self):
        descriptors = torch.eye(10).unsqueeze(0)
        correct = attachment.cross_channel_attachment_loss(descriptors, descriptors)
        expected = math.log(math.e + 9.0) - 1.0
        self.assertAlmostEqual(correct.item(), expected, places=6)

        permutation = torch.roll(descriptors, shifts=1, dims=1)
        permuted = attachment.cross_channel_attachment_loss(
            descriptors,
            permutation,
        )
        self.assertAlmostEqual(permuted.item(), math.log(math.e + 9.0), places=6)
        self.assertGreater(permuted.item(), correct.item())

        logits = torch.einsum("bic,bjc->bij", descriptors, permutation)
        labels = torch.arange(10).expand(1, 10)
        manual = F.cross_entropy(logits.reshape(10, 10), labels.reshape(10))
        torch.testing.assert_close(permuted, manual, rtol=0.0, atol=0.0)

    def test_identical_zero_and_near_zero_are_finite_log_ten(self):
        identical = torch.ones(2, 10, 4)
        loss = attachment.cross_channel_attachment_loss(identical, identical)
        self.assertAlmostEqual(loss.item(), math.log(10.0), places=6)
        for magnitude in (0.0, 1e-12):
            descriptors = torch.full((1, 10, 8), magnitude)
            observed = attachment.cross_channel_attachment_loss(
                descriptors,
                descriptors,
            )
            self.assertTrue(torch.isfinite(observed))
            self.assertAlmostEqual(observed.item(), math.log(10.0), places=6)

    def test_symmetric_loss_is_frame_swap_invariant(self):
        torch.manual_seed(9)
        f_t = _exact_features()
        f_t1 = _exact_features()
        p_t = _coordinates()
        p_t1 = torch.flip(_coordinates(), dims=(1,))
        forward = attachment.symmetric_descriptor_attachment_loss(
            f_t, p_t, f_t1, p_t1
        )
        swapped = attachment.symmetric_descriptor_attachment_loss(
            f_t1, p_t1, f_t, p_t
        )
        torch.testing.assert_close(forward, swapped, rtol=0.0, atol=1e-7)

    def test_constant_map_and_duplicates_expose_stationary_limitations(self):
        f_t = _exact_features(requires_grad=True, constant=1.0)
        f_t1 = _exact_features(requires_grad=True, constant=1.0)
        p_t = _coordinates(requires_grad=True)
        p_t1 = _coordinates(requires_grad=True)
        loss = attachment.descriptor_attachment_one_way(f_t, p_t, f_t1, p_t1)
        self.assertAlmostEqual(loss.item(), math.log(10.0), places=5)
        candidate_gradient = torch.autograd.grad(loss, p_t1)[0]
        torch.testing.assert_close(
            candidate_gradient,
            torch.zeros_like(candidate_gradient),
            rtol=0.0,
            atol=1e-7,
        )

        anchor = torch.ones(1, 10, 4)
        candidate = torch.ones(1, 10, 4, requires_grad=True)
        duplicate_loss = attachment.cross_channel_attachment_loss(anchor, candidate)
        duplicate_gradient = torch.autograd.grad(duplicate_loss, candidate)[0]
        torch.testing.assert_close(
            duplicate_gradient,
            torch.zeros_like(duplicate_gradient),
            rtol=0.0,
            atol=1e-7,
        )


class SamplingAndGradientTests(unittest.TestCase):
    def test_xy_corners_interior_values_and_coordinate_gradient(self):
        feature_map = torch.tensor(
            [[[[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]]]],
            requires_grad=True,
        )
        corners = torch.tensor(
            [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]
        )
        sampled = attachment.sample_feature_vectors(
            feature_map,
            corners,
            detach_features=False,
            detach_coordinates=False,
        )
        torch.testing.assert_close(
            sampled.squeeze(-1),
            torch.tensor([[0.0, 4.0, 10.0, 14.0]]),
            rtol=0.0,
            atol=0.0,
        )

        interior = torch.tensor([[[0.5, 0.0]]], requires_grad=True)
        interior_value = attachment.sample_feature_vectors(
            feature_map,
            interior,
            detach_features=False,
            detach_coordinates=False,
        )
        # align_corners=True maps x=.5 to column 1.5 and y=0 to row .5.
        self.assertAlmostEqual(interior_value.item(), 8.0, places=6)
        coordinate_gradient = torch.autograd.grad(interior_value.sum(), interior)[0]
        torch.testing.assert_close(
            coordinate_gradient,
            torch.tensor([[[2.0, 5.0]]]),
            rtol=0.0,
            atol=1e-6,
        )

    def test_strict_configuration_and_shape_fail_closed(self):
        attachment.validate_retained_runtime_configuration(
            img_size=512,
            heatmap_res=64,
            base_channels=32,
            num_keypoints=10,
        )
        for changed in (
            {"img_size": 256},
            {"heatmap_res": 128},
            {"base_channels": 16},
            {"num_keypoints": 9},
        ):
            config = {
                "img_size": 512,
                "heatmap_res": 64,
                "base_channels": 32,
                "num_keypoints": 10,
                **changed,
            }
            with self.assertRaises(ValueError):
                attachment.validate_retained_runtime_configuration(**config)
        with self.assertRaises(ValueError):
            attachment.symmetric_descriptor_attachment_loss(
                torch.randn(1, 64, 64, 64),
                _coordinates(),
                torch.randn(1, 64, 64, 64),
                _coordinates(),
            )

    def test_one_way_direct_gradient_isolation(self):
        torch.manual_seed(11)
        f_anchor = _exact_features(requires_grad=True)
        f_candidate = _exact_features(requires_grad=True)
        p_anchor = _coordinates(requires_grad=True)
        p_candidate = (_coordinates() + 0.013).requires_grad_(True)
        loss = attachment.descriptor_attachment_one_way(
            f_anchor,
            p_anchor,
            f_candidate,
            p_candidate,
        )
        gradients = torch.autograd.grad(
            loss,
            (f_anchor, p_anchor, f_candidate, p_candidate),
            allow_unused=True,
        )
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        self.assertIsNotNone(gradients[3])
        self.assertGreater(float(gradients[3].abs().sum()), 0.0)

    def test_symmetric_gradient_reaches_both_coordinate_paths(self):
        torch.manual_seed(17)
        p_t = _coordinates(requires_grad=True)
        p_t1 = (_coordinates() + 0.019).requires_grad_(True)
        loss = attachment.symmetric_descriptor_attachment_loss(
            _exact_features(), p_t, _exact_features(), p_t1
        )
        gradient_t, gradient_t1 = torch.autograd.grad(loss, (p_t, p_t1))
        self.assertGreater(float(gradient_t.abs().sum()), 0.0)
        self.assertGreater(float(gradient_t1.abs().sum()), 0.0)

    def test_integrated_gradient_reaches_encoder_and_head_only(self):
        torch.manual_seed(23)
        model = model_module.PhaseAModel(
            num_keypoints=10,
            base_channels=32,
            num_action_classes=2,
            operator_type="shared_affine",
            learn_inverse_operator=True,
            heatmap_res=64,
        )
        outputs = model(
            torch.randn(1, 3, 512, 512),
            torch.randn(1, 3, 512, 512),
            return_descriptor_features=True,
        )
        loss = attachment.symmetric_descriptor_attachment_loss(
            outputs["descriptor_features_t"],
            outputs["p_t"].view(1, 10, 2),
            outputs["descriptor_features_t1"],
            outputs["p_t1"].view(1, 10, 2),
        )
        loss.backward()
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.extractor.encoder.parameters()
            if parameter.grad is not None
        )
        head_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.extractor.heatmap_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)
        self.assertGreater(head_grad, 0.0)
        for module in (
            model.operator,
            model.inverse_operator,
            model.action_classifier,
        ):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))


class IntegrationAndZeroArmTests(unittest.TestCase):
    def test_attachment_reuses_two_forwards_and_bn_updates_match_zero_arm(self):
        torch.manual_seed(29)
        zero_model = model_module.PhaseAModel(heatmap_res=64)
        attachment_model = model_module.PhaseAModel(heatmap_res=64)
        attachment_model.load_state_dict(zero_model.state_dict())
        zero_model.train()
        attachment_model.train()
        x_t = torch.randn(1, 3, 512, 512)
        x_t1 = torch.randn(1, 3, 512, 512)
        counts = {"zero": 0, "attachment": 0}
        h0 = zero_model.extractor.encoder.register_forward_hook(
            lambda *_: counts.__setitem__("zero", counts["zero"] + 1)
        )
        h1 = attachment_model.extractor.encoder.register_forward_hook(
            lambda *_: counts.__setitem__("attachment", counts["attachment"] + 1)
        )
        zero_model(x_t, x_t1)
        attachment_model(x_t, x_t1, return_descriptor_features=True)
        h0.remove()
        h1.remove()
        self.assertEqual(counts, {"zero": 2, "attachment": 2})
        zero_bn = {
            name: value
            for name, value in zero_model.state_dict().items()
            if "running_" in name or "num_batches_tracked" in name
        }
        attach_bn = {
            name: value
            for name, value in attachment_model.state_dict().items()
            if "running_" in name or "num_batches_tracked" in name
        }
        self.assertEqual(set(zero_bn), set(attach_bn))
        for name in zero_bn:
            torch.testing.assert_close(zero_bn[name], attach_bn[name], rtol=0.0, atol=0.0)

    def test_zero_arm_never_calls_attachment_and_preserves_outputs(self):
        torch.manual_seed(31)
        model = model_module.PhaseAModel()
        x_t = torch.randn(1, 3, 64, 64)
        x_t1 = torch.randn(1, 3, 64, 64)
        first = model(x_t, x_t1)
        second = model(x_t, x_t1, return_descriptor_features=False)
        self.assertEqual(set(first), set(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
        with mock.patch.object(
            model_module,
            "symmetric_descriptor_attachment_loss",
            side_effect=AssertionError("zero arm entered descriptor code"),
        ):
            losses = model_module.compute_losses(first, lambda_attach=0.0)
        self.assertIs(losses["loss"], losses["base_loss"])
        self.assertEqual(losses["attachment_loss"].item(), 0.0)

    def test_zero_arm_one_step_optimizer_scheduler_and_old_state_loading(self):
        torch.manual_seed(37)
        left = model_module.PhaseAModel()
        right = model_module.PhaseAModel()
        right.load_state_dict(left.state_dict(), strict=True)
        left_optimizer = torch.optim.Adam(left.parameters(), lr=1e-4)
        right_optimizer = torch.optim.Adam(right.parameters(), lr=1e-4)
        left_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(left_optimizer, T_max=2)
        right_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(right_optimizer, T_max=2)
        x_t = torch.randn(1, 3, 64, 64)
        x_t1 = torch.randn(1, 3, 64, 64)
        for model, optimizer, scheduler, explicit in (
            (left, left_optimizer, left_scheduler, False),
            (right, right_optimizer, right_scheduler, True),
        ):
            outputs = model(x_t, x_t1)
            kwargs = {"lambda_attach": 0.0} if explicit else {}
            loss = model_module.compute_losses(outputs, **kwargs)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        for name, tensor in left.state_dict().items():
            torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0.0, atol=0.0)
        left_optimizer_state = left_optimizer.state_dict()
        right_optimizer_state = right_optimizer.state_dict()
        self.assertEqual(
            left_optimizer_state["param_groups"],
            right_optimizer_state["param_groups"],
        )
        self.assertEqual(
            set(left_optimizer_state["state"]),
            set(right_optimizer_state["state"]),
        )
        for parameter_id, left_state in left_optimizer_state["state"].items():
            right_state = right_optimizer_state["state"][parameter_id]
            self.assertEqual(set(left_state), set(right_state))
            for key, left_value in left_state.items():
                right_value = right_state[key]
                if torch.is_tensor(left_value):
                    torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)
                else:
                    self.assertEqual(left_value, right_value)
        self.assertEqual(left_scheduler.state_dict(), right_scheduler.state_dict())

        checkpoint = Path(tempfile.mkdtemp(prefix="descriptor_old_checkpoint_")) / "old.pt"
        torch.save({"model_state_dict": left.state_dict()}, checkpoint)
        loaded = model_module.PhaseAModel()
        loaded.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"], strict=True)

    def test_base_checkpoint_selector_ignores_reversed_augmented_order(self):
        epochs = [
            {"epoch": 1, "base": 1.0, "augmented": 1.1},
            {"epoch": 2, "base": 0.9, "augmented": 2.0},
        ]
        best_loss = float("inf")
        best_epoch = None
        for record in epochs:
            if train_module.base_validation_improves(record["base"], best_loss):
                best_loss = record["base"]
                best_epoch = record["epoch"]
        self.assertEqual(best_epoch, 2)
        self.assertEqual(min(epochs, key=lambda record: record["augmented"])["epoch"], 1)
        self.assertFalse(train_module.base_validation_improves(best_loss, best_loss))

    def test_operator_outputs_do_not_change_attachment_component(self):
        torch.manual_seed(41)
        p_t = _coordinates().reshape(1, 20)
        p_t1 = (_coordinates() + 0.01).reshape(1, 20)
        common = {
            "p_t": p_t,
            "p_t1": p_t1,
            "heatmaps_t": torch.randn(1, 10, 64, 64),
            "heatmaps_t1": torch.randn(1, 10, 64, 64),
            "descriptor_features_t": _exact_features(),
            "descriptor_features_t1": _exact_features(),
        }
        first = model_module.compute_losses(
            {**common, "p_hat_t1": torch.zeros_like(p_t1)},
            lambda_attach=1.0,
        )
        second = model_module.compute_losses(
            {**common, "p_hat_t1": torch.full_like(p_t1, 7.0)},
            lambda_attach=1.0,
        )
        torch.testing.assert_close(
            first["attachment_loss"],
            second["attachment_loss"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotEqual(first["base_loss"].item(), second["base_loss"].item())


if __name__ == "__main__":
    unittest.main()
