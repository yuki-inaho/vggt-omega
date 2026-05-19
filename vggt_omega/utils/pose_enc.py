# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from jaxtyping import Float

from .rotation import mat_to_quat, quat_to_mat


def extri_intri_to_pose_encoding(
    extrinsics: Float[torch.Tensor, "b n_img 3 4"],
    intrinsics: Float[torch.Tensor, "b n_img 3 3"],
    image_size_hw: tuple[int, int],
) -> Float[torch.Tensor, "b n_img 9"]:
    """Convert camera extrinsics and intrinsics to VGGT-Omega pose encoding.

    The released checkpoints use a 9D camera encoding:
    translation (3), quaternion rotation (4), and vertical/horizontal FoV (2).
    Extrinsics are camera-from-world matrices in OpenCV coordinates.
    """
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]

    H, W = image_size_hw
    quat = mat_to_quat(R)
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def encoding_to_camera(
    pose_encoding: Float[torch.Tensor, "b n_img 9"],
    image_size_hw: tuple[int, int],
    build_intrinsics: bool = True,
) -> tuple[Float[torch.Tensor, "b n_img 3 4"], Float[torch.Tensor, "b n_img 3 3"] | None]:
    """Decode VGGT-Omega pose encoding into extrinsics and intrinsics."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics
