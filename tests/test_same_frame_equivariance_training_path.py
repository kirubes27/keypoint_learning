"""Production-path guards for the paired self-equivariance GPU smoke."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
KEYPOINT_NET_ROOT = REPO_ROOT / "keypoint_net"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KEYPOINT_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_NET_ROOT))

from keypoint_net import train as train_module
from keypoint_net import run_same_frame_equivariance_smoke as smoke_runner
from keypoint_net import (
    same_frame_equivariance_smoke_authorization as smoke_authorization,
)


class _MarkerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(1.0))
        self.extractor = nn.Identity()

    def forward(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor,
        *,
        return_descriptor_features: bool,
    ) -> dict[str, torch.Tensor]:
        del x_t1, return_descriptor_features
        value = self.marker * x_t.mean()
        batch_size = int(x_t.shape[0])
        heatmaps = value.expand(batch_size, 10, 2, 2)
        return {"marker": value, "heatmaps_t": heatmaps, "heatmaps_t1": heatmaps}


def _losses(outputs: dict[str, torch.Tensor], **_: object) -> dict[str, torch.Tensor]:
    marker = outputs["marker"]
    zero = marker * 0.0
    return {
        "loss": marker,
        "base_loss": marker,
        "attachment_loss": zero,
        "l_pred": marker,
        "l_smooth": zero,
        "l_disp": zero,
        "l_ent": zero,
        "l_act": zero,
        "l_loc": zero,
        "l_inv": zero,
        "l_cycle": zero,
        "act_acc": zero,
    }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "x_t": torch.ones(2, 1),
        "x_t1": torch.zeros(2, 1),
        "action_label": torch.zeros(2, dtype=torch.long),
        "pair_id": ["a", "b"],
    }


def _paired_result(base_loss: torch.Tensor, weight: float) -> types.SimpleNamespace:
    auxiliary = base_loss * 0.25
    components = {
        "translation_plus_quarter_pixel_xy": auxiliary * 0.8,
        "rotation_plus_quarter_degree_clockwise_image": auxiliary * 1.2,
    }
    optimization = base_loss if weight == 0.0 else base_loss + weight * auxiliary
    return types.SimpleNamespace(
        optimization_loss=optimization,
        equivariance=types.SimpleNamespace(
            loss=auxiliary,
            per_transform_loss=components,
        ),
    )


class TrainingPathTest(unittest.TestCase):
    def _run_train(self, weight: float) -> tuple[dict, mock.Mock]:
        model = _MarkerModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def paired(*args: object, **kwargs: object) -> types.SimpleNamespace:
            return _paired_result(args[5], float(kwargs["weight"]))

        paired_mock = mock.Mock(side_effect=paired)
        with (
            mock.patch.object(train_module, "compute_losses", side_effect=_losses),
            mock.patch.object(
                train_module,
                "add_locked_equivariance_to_pair_loss",
                paired_mock,
            ),
        ):
            result = train_module.train_epoch(
                model,
                [_batch()],
                optimizer,
                torch.device("cpu"),
                lambda_smooth=0.0,
                lambda_disp=0.1,
                lambda_ent=0.01,
                lambda_act=0.0,
                lambda_loc=0.0,
                lambda_inv=0.0,
                lambda_cycle=0.0,
                loc_bg_threshold=30.0,
                sigma=0.1,
                num_keypoints=10,
                execute_self_equivariance=True,
                lambda_self_equivariance=weight,
                record_pair_order=True,
            )
        return result, paired_mock

    def test_zero_control_executes_same_helper_and_keeps_base_loss(self) -> None:
        result, paired_mock = self._run_train(0.0)
        self.assertEqual(paired_mock.call_count, 1)
        self.assertEqual(paired_mock.call_args.kwargs["weight"], 0.0)
        self.assertEqual(result["loss"], result["base_loss"])
        self.assertGreater(result["self_equivariance_loss"], 0.0)
        self.assertEqual(result["pair_order_sample_count"], 2)

    def test_manifest_arguments_differ_only_by_candidate_weight(self) -> None:
        common = {
            "recipe": "task80_assisted",
            "data_root": Path("/dataset"),
            "train_index": Path("/train.json"),
            "validation_index": Path("/validation.json"),
            "output_root": Path("/output"),
            "batch_size": 4,
        }
        control = smoke_runner._arguments(arm="control", **common)  # noqa: SLF001
        candidate = smoke_runner._arguments(  # noqa: SLF001
            arm="self_equivariance", **common
        )
        differences = {
            name: (control[name], candidate[name])
            for name in control
            if control[name] != candidate[name]
        }
        self.assertEqual(
            differences,
            {
                "lambda_self_equivariance": (
                    0.0,
                    smoke_authorization.CALIBRATED_COEFFICIENT,
                )
            },
        )
        self.assertEqual(set(control), smoke_authorization.BOUND_ARGUMENTS)

    def test_candidate_applies_only_the_locked_weight(self) -> None:
        result, paired_mock = self._run_train(0.5)
        self.assertEqual(paired_mock.call_count, 1)
        self.assertEqual(paired_mock.call_args.kwargs["weight"], 0.5)
        self.assertAlmostEqual(
            result["loss"],
            result["base_loss"] + 0.5 * result["self_equivariance_loss"],
            places=6,
        )

    def test_validation_control_also_executes_common_path(self) -> None:
        model = _MarkerModel()

        def paired(*args: object, **kwargs: object) -> types.SimpleNamespace:
            return _paired_result(args[5], float(kwargs["weight"]))

        paired_mock = mock.Mock(side_effect=paired)
        with (
            mock.patch.object(train_module, "compute_losses", side_effect=_losses),
            mock.patch.object(
                train_module,
                "add_locked_equivariance_to_pair_loss",
                paired_mock,
            ),
        ):
            result = train_module.evaluate(
                model,
                [_batch()],
                torch.device("cpu"),
                lambda_smooth=0.0,
                lambda_disp=0.1,
                lambda_ent=0.01,
                lambda_act=0.0,
                lambda_loc=0.0,
                lambda_inv=0.0,
                lambda_cycle=0.0,
                loc_bg_threshold=30.0,
                sigma=0.1,
                num_keypoints=10,
                execute_self_equivariance=True,
                lambda_self_equivariance=0.0,
            )
        self.assertEqual(paired_mock.call_count, 1)
        self.assertEqual(result["loss"], result["base_loss"])
        self.assertGreater(result["self_equivariance_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
