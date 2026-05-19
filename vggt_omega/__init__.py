# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega inference package."""

from .inference import InferenceResult, VGGTOmegaInference
from .models import VGGTOmega
from .pipeline import SceneResult, VGGTOmegaPipeline

__version__ = "0.0.1"

__all__ = [
    "InferenceResult",
    "SceneResult",
    "VGGTOmega",
    "VGGTOmegaInference",
    "VGGTOmegaPipeline",
    "__version__",
]
