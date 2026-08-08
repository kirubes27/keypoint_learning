"""Exact learned descriptor-attachment intervention for the retained roll head.

This module implements the reviewed intervention named:

    symmetric stop-gradient within-pair cross-channel learned
    descriptor-attachment loss

It tests local-appearance attachment with jointly learned encoder descriptors
and cross-channel negatives.  It is not literal brightness-normalized RGB
patch MSE.  The frozen specification below is code-native so authorization
artifacts never depend on prose documentation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn.functional as F


LOSS_SPEC: dict[str, Any] = {
    "schema_version": "descriptor_attachment_loss_spec.v1",
    "name": (
        "symmetric stop-gradient within-pair cross-channel learned "
        "descriptor-attachment loss"
    ),
    "scientific_scope": {
        "tests_local_appearance_attachment": True,
        "learned_encoder_descriptor": True,
        "literal_brightness_normalized_rgb_patch_mse": False,
        "cross_channel_negatives_are_additional_refinement": True,
    },
    "retained_configuration": {
        "img_size": 512,
        "heatmap_res": 64,
        "base_channels": 32,
        "num_keypoints": 10,
        "descriptor_feature_shape_without_batch": [128, 64, 64],
        "descriptor_location": "post_relu_encoder_pre_1x1_heatmap_head",
    },
    "sampling": {
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": True,
        "coordinate_order": "xy",
        "coordinate_range": [-1.0, 1.0],
        "normalization": "l2",
        "normalization_epsilon": 1e-6,
    },
    "contrast": {
        "channels": 10,
        "positive": "same_channel_within_pair",
        "negatives": "other_9_target_channels_within_pair",
        "logit_scale": 1.0,
        "temperature": None,
        "projection_head": None,
        "in_batch_negatives": False,
        "random_spatial_negatives": False,
    },
    "gradient_semantics": {
        "anchor_feature_values": "stop_gradient",
        "anchor_coordinates": "stop_gradient",
        "candidate_feature_values": "stop_gradient",
        "candidate_coordinates": "differentiable",
        "directions": ["t_to_t1", "t1_to_t"],
        "direction_reduction": "mean",
    },
    "forbidden_inputs": [
        "predicted_next_keypoints",
        "operator_parameters",
        "inverse_operator_parameters",
        "action_head_parameters",
        "masks",
        "action_metadata",
        "transform_metadata",
        "evaluator_outputs",
    ],
    "checkpoint_selector": "minimum_base_validation_loss",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used by all bound artifacts."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


LOSS_SPEC_SHA256 = canonical_sha256(LOSS_SPEC)
NORMALIZATION_EPSILON = 1e-6
LOGIT_SCALE = 1.0
NUM_KEYPOINTS = 10
DESCRIPTOR_CHANNELS = 128
DESCRIPTOR_HEIGHT = 64
DESCRIPTOR_WIDTH = 64


def validate_retained_runtime_configuration(
    *,
    img_size: int,
    heatmap_res: int,
    base_channels: int,
    num_keypoints: int,
) -> None:
    """Fail before a positive-arm forward if the retained package changed."""

    observed = {
        "img_size": img_size,
        "heatmap_res": heatmap_res,
        "base_channels": base_channels,
        "num_keypoints": num_keypoints,
    }
    expected = {
        "img_size": 512,
        "heatmap_res": 64,
        "base_channels": 32,
        "num_keypoints": 10,
    }
    if observed != expected:
        raise ValueError(
            "descriptor attachment requires the exact retained configuration; "
            f"expected {expected}, observed {observed}"
        )


def _validate_sampling_inputs(
    features: torch.Tensor,
    coordinates: torch.Tensor,
) -> None:
    if features.ndim != 4:
        raise ValueError("descriptor features must have shape (B,C,H,W)")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("descriptor coordinates must have shape (B,N,2)")
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError("descriptor feature/coordinate batch sizes differ")
    if not features.is_floating_point() or not coordinates.is_floating_point():
        raise TypeError("descriptor features and coordinates must be floating point")
    if features.device != coordinates.device:
        raise ValueError("descriptor features and coordinates must share a device")


def sample_feature_vectors(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    detach_features: bool,
    detach_coordinates: bool,
) -> torch.Tensor:
    """Sample raw feature vectors at normalized ``(x,y)`` coordinates.

    The output has shape ``(B,N,C)``.  This lower-level function is public so
    the exact corner, interpolation, coordinate-order, and gradient semantics
    can be tested independently from descriptor normalization and contrast.
    """

    _validate_sampling_inputs(features, coordinates)
    feature_values = features.detach() if detach_features else features
    grid = coordinates.detach() if detach_coordinates else coordinates
    sampled = F.grid_sample(
        feature_values,
        grid.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2)


def sample_normalized_descriptors(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    detach_coordinates: bool,
) -> torch.Tensor:
    """Sample stop-gradient feature values and L2-normalize with epsilon 1e-6."""

    sampled = sample_feature_vectors(
        features,
        coordinates,
        detach_features=True,
        detach_coordinates=detach_coordinates,
    )
    return F.normalize(sampled, p=2.0, dim=-1, eps=NORMALIZATION_EPSILON)


def cross_channel_attachment_loss(
    anchor_descriptors: torch.Tensor,
    candidate_descriptors: torch.Tensor,
) -> torch.Tensor:
    """Mean same-channel cross entropy over batch and anchor channel."""

    if anchor_descriptors.ndim != 3 or candidate_descriptors.ndim != 3:
        raise ValueError("descriptors must have shape (B,N,C)")
    if anchor_descriptors.shape != candidate_descriptors.shape:
        raise ValueError("anchor and candidate descriptor shapes differ")
    batch, channels, _ = anchor_descriptors.shape
    if channels != NUM_KEYPOINTS:
        raise ValueError(f"descriptor attachment requires exactly {NUM_KEYPOINTS} channels")
    logits = torch.einsum(
        "bic,bjc->bij", anchor_descriptors, candidate_descriptors
    ) * LOGIT_SCALE
    labels = torch.arange(channels, device=logits.device).expand(batch, channels)
    return F.cross_entropy(
        logits.reshape(batch * channels, channels),
        labels.reshape(batch * channels),
        reduction="mean",
    )


def _validate_retained_descriptor_tensors(
    features_t: torch.Tensor,
    coordinates_t: torch.Tensor,
    features_t1: torch.Tensor,
    coordinates_t1: torch.Tensor,
) -> None:
    _validate_sampling_inputs(features_t, coordinates_t)
    _validate_sampling_inputs(features_t1, coordinates_t1)
    if features_t.shape != features_t1.shape:
        raise ValueError("paired descriptor feature shapes differ")
    if coordinates_t.shape != coordinates_t1.shape:
        raise ValueError("paired descriptor coordinate shapes differ")
    expected_features = (
        features_t.shape[0],
        DESCRIPTOR_CHANNELS,
        DESCRIPTOR_HEIGHT,
        DESCRIPTOR_WIDTH,
    )
    if tuple(features_t.shape) != expected_features:
        raise ValueError(
            "descriptor features must have exact retained shape "
            f"{expected_features}, observed {tuple(features_t.shape)}"
        )
    expected_coordinates = (features_t.shape[0], NUM_KEYPOINTS, 2)
    if tuple(coordinates_t.shape) != expected_coordinates:
        raise ValueError(
            "descriptor coordinates must have exact retained shape "
            f"{expected_coordinates}, observed {tuple(coordinates_t.shape)}"
        )


def descriptor_attachment_one_way(
    anchor_features: torch.Tensor,
    anchor_coordinates: torch.Tensor,
    candidate_features: torch.Tensor,
    candidate_coordinates: torch.Tensor,
) -> torch.Tensor:
    """One reviewed teacher-to-candidate direction of attachment contrast."""

    _validate_retained_descriptor_tensors(
        anchor_features,
        anchor_coordinates,
        candidate_features,
        candidate_coordinates,
    )
    anchors = sample_normalized_descriptors(
        anchor_features,
        anchor_coordinates,
        detach_coordinates=True,
    )
    candidates = sample_normalized_descriptors(
        candidate_features,
        candidate_coordinates,
        detach_coordinates=False,
    )
    return cross_channel_attachment_loss(anchors, candidates)


def symmetric_descriptor_attachment_loss(
    features_t: torch.Tensor,
    coordinates_t: torch.Tensor,
    features_t1: torch.Tensor,
    coordinates_t1: torch.Tensor,
) -> torch.Tensor:
    """Mean of the two reviewed stop-gradient within-pair directions."""

    forward = descriptor_attachment_one_way(
        features_t,
        coordinates_t,
        features_t1,
        coordinates_t1,
    )
    reverse = descriptor_attachment_one_way(
        features_t1,
        coordinates_t1,
        features_t,
        coordinates_t,
    )
    return 0.5 * (forward + reverse)


__all__ = [
    "DESCRIPTOR_CHANNELS",
    "DESCRIPTOR_HEIGHT",
    "DESCRIPTOR_WIDTH",
    "LOGIT_SCALE",
    "LOSS_SPEC",
    "LOSS_SPEC_SHA256",
    "NORMALIZATION_EPSILON",
    "NUM_KEYPOINTS",
    "canonical_json_bytes",
    "canonical_sha256",
    "cross_channel_attachment_loss",
    "descriptor_attachment_one_way",
    "sample_feature_vectors",
    "sample_normalized_descriptors",
    "symmetric_descriptor_attachment_loss",
    "validate_retained_runtime_configuration",
]
