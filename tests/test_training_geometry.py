from __future__ import annotations

import pytest
import torch

from vggt_omega.training.geometry import (
    GeometryContractError,
    camera_to_world_to_world_to_camera,
    normalize_supervision,
    project_points,
    unproject_depth,
)


def test_camera_to_world_to_world_to_camera_known_transform() -> None:
    camera_to_world = torch.tensor(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    world_to_camera = camera_to_world_to_world_to_camera(camera_to_world)

    assert world_to_camera.shape == (3, 4)
    homogeneous = torch.eye(4, dtype=torch.float64)
    homogeneous[:3] = world_to_camera
    assert torch.allclose(homogeneous @ camera_to_world, torch.eye(4, dtype=torch.float64), atol=1e-10)


def test_unproject_project_round_trip() -> None:
    depth = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=torch.float64)
    intrinsics = torch.tensor([[4.0, 0.0, 0.75], [0.0, 5.0, 0.25], [0.0, 0.0, 1.0]], dtype=torch.float64)

    camera_points = unproject_depth(depth, intrinsics)
    pixels, projected_depth = project_points(camera_points, intrinsics)
    yy, xx = torch.meshgrid(torch.arange(2, dtype=torch.float64), torch.arange(3, dtype=torch.float64), indexing="ij")

    assert torch.allclose(pixels, torch.stack((xx, yy), dim=-1), atol=1e-10)
    assert torch.allclose(projected_depth, depth, atol=1e-10)


def test_normalize_supervision_uses_first_camera_and_one_common_scale() -> None:
    depths = torch.tensor(
        [
            [[2.0, 2.0], [2.0, 2.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ],
        dtype=torch.float64,
    )
    masks = torch.ones_like(depths, dtype=torch.bool)
    intrinsics = torch.tensor(
        [
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    extrinsics = torch.tensor(
        [
            [[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0, -5.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    result = normalize_supervision(depths, masks, intrinsics, extrinsics)

    expected_identity = torch.cat((torch.eye(3, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)), 1)
    assert torch.allclose(result["extrinsics"][0], expected_identity, atol=1e-10)
    assert torch.allclose(result["intrinsics"], intrinsics)
    assert torch.allclose(result["extrinsics"][:, :3, :3], extrinsics[:, :3, :3])
    assert torch.allclose(result["depths"] * result["scale"], depths)
    assert torch.allclose(
        result["extrinsics"][1, :, 3] * result["scale"],
        torch.tensor([-3.0, 0.0, 0.0], dtype=torch.float64),
    )
    valid_points = result["world_points"][result["depth_masks"]]
    assert torch.linalg.vector_norm(valid_points, dim=-1).mean().item() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    "failure",
    ["singular_intrinsics", "empty_mask", "singular_pose", "non_rigid_pose", "non_finite_pose"],
)
def test_geometry_contract_errors_are_explicit(failure: str) -> None:
    depths = torch.ones((1, 2, 2), dtype=torch.float32)
    masks = torch.ones_like(depths, dtype=torch.bool)
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    extrinsics = torch.eye(4, dtype=torch.float32)[:3].unsqueeze(0)
    if failure == "singular_intrinsics":
        intrinsics[0, 2, 2] = 0
    elif failure == "empty_mask":
        masks[:] = False
    elif failure == "singular_pose":
        extrinsics[0, :3, :3] = 0
    elif failure == "non_rigid_pose":
        extrinsics[0, 0, 0] = 2
    else:
        extrinsics[0, 0, 0] = torch.nan

    with pytest.raises(GeometryContractError):
        normalize_supervision(depths, masks, intrinsics, extrinsics)
