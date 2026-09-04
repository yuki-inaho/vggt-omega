from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import DataLoader

from scripts.prepare_colmap_rgbd_training import (
    ExportContractError,
    compute_principal_crop,
    export_colmap_rgbd_training,
    main,
    prepare_overlap_profile,
    validate_staging,
)
from vggt_omega.training.dataset import (
    ColmapRgbdDataset,
    DataContractError,
    overlap_curriculum_target,
    select_overlap_frame_offsets,
)

PRIVATE_TOKEN = "private-session-7a7e56ff-customer"


def _write_private_source(tmp_path: Path, frame_count: int = 15) -> tuple[Path, Path, Path]:
    source = tmp_path / PRIVATE_TOKEN
    rgb_dir = source / "rgb"
    depth_dir = source / "mapped_depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()
    frames = []
    for index in range(frame_count):
        basename = f"{PRIVATE_TOKEN}__2026-09-03_{index:06d}"
        rgb_name = f"{basename}_rgb.png"
        depth_name = f"{basename}_depth.png"
        rgb = np.full((48, 64, 3), 20 + index, dtype=np.uint8)
        rgb_info = PngImagePlugin.PngInfo()
        rgb_info.add_text("source", f"/private/{PRIVATE_TOKEN}/{rgb_name}")
        Image.fromarray(rgb, mode="RGB").save(rgb_dir / rgb_name, pnginfo=rgb_info)
        depth = np.zeros((48, 64), dtype=np.uint16)
        depth[24, 32] = 800 + index
        Image.fromarray(depth).save(depth_dir / depth_name)
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[0, 3] = index * 0.01
        frames.append(
            {
                "frame_index": index,
                "image_name": rgb_name,
                "path": f"/private/{PRIVATE_TOKEN}/{rgb_name}",
                "camera_to_world": camera_to_world.tolist(),
            }
        )
    (source / "README.md").write_text(
        "mapped_depth was generated with generate_mapped_depth.py and 3x3 nearest-depth dilation.\n"
    )
    trajectory = source / "private_trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "pose_convention": "camera_to_world_tum_xyz_qw_qx_qy_qz",
                "frame_count": frame_count,
                "frames": frames,
                "chunk_scales": [{"chunk_index": 0, "global_indices": list(range(frame_count))}],
            }
        )
    )
    camera_yaml = source / "private_camera.yaml"
    camera_yaml.write_text(
        "\n".join(
            [
                f"serial_number: {PRIVATE_TOKEN}",
                "height: 48",
                "width: 64",
                "K: [50.0, 0.0, 32.0, 0.0, 51.0, 24.0, 0.0, 0.0, 1.0]",
            ]
        )
    )
    return source, trajectory, camera_yaml


@pytest.fixture
def exported_staging(tmp_path: Path) -> Path:
    source, trajectory, camera_yaml = _write_private_source(tmp_path)
    output = tmp_path / "colmap_rgbd_v1"
    report = export_colmap_rgbd_training(
        source_rgbd_root=source,
        trajectory_path=trajectory,
        rgb_camera_yaml=camera_yaml,
        output_root=output,
        height=48,
        width=64,
        expected_dilation_kernel=3,
        guard_frames=1,
        split_fractions=(0.6, 0.2, 0.2),
    )
    assert report["valid"]
    return output


def test_compute_principal_crop_centers_principal_point_and_updates_focal_length() -> None:
    intrinsics = np.array([[50.0, 0.0, 35.0], [0.0, 52.0, 28.0], [0.0, 0.0, 1.0]])

    crop, updated = compute_principal_crop(intrinsics, source_size=(80, 60), target_size=(64, 48))

    x0, y0, x1, y1 = crop
    assert (x1 - x0) / (y1 - y0) == pytest.approx(64 / 48)
    assert updated[0, 2] == pytest.approx(32.0)
    assert updated[1, 2] == pytest.approx(24.0)
    assert updated[0, 0] == pytest.approx(50.0 * 64 / (x1 - x0))
    assert updated[1, 1] == pytest.approx(52.0 * 48 / (y1 - y0))


def test_exporter_reencodes_private_source_to_generic_regular_files(exported_staging: Path) -> None:
    paths = [path for path in exported_staging.rglob("*") if path.is_file()]
    relative_names = [path.relative_to(exported_staging).as_posix() for path in paths]

    assert not any(path.is_symlink() for path in exported_staging.rglob("*"))
    assert all(PRIVATE_TOKEN not in name for name in relative_names)
    assert "scenes/scene_000000/rgb/frame_000000.png" in relative_names
    assert "scenes/scene_000000/depth/frame_000000.png" in relative_names
    assert all(PRIVATE_TOKEN.encode() not in path.read_bytes() for path in paths)
    for path in exported_staging.rglob("*.png"):
        with Image.open(path) as image:
            assert image.info == {}
    with np.load(exported_staging / "scenes/scene_000000/cameras.npz", allow_pickle=False) as cameras:
        assert cameras["frame_ids"].tolist() == list(range(15))
        assert cameras["intrinsics"].shape == (15, 3, 3)
        assert cameras["extrinsics_w2c"].shape == (15, 3, 4)
        assert cameras["intrinsics"][0, 0, 2] == pytest.approx(32.0)
        assert cameras["extrinsics_w2c"][4, 0, 3] == pytest.approx(-0.04)
    exported_depth = np.asarray(Image.open(exported_staging / "scenes/scene_000000/depth/frame_000000.png"))
    assert np.count_nonzero(exported_depth) == 1, "export must not apply a second dilation"


def test_exporter_writes_anonymous_numeric_overlap_profile(exported_staging: Path) -> None:
    metadata = json.loads((exported_staging / "dataset.json").read_text())
    assert metadata["overlap"] == {
        "aggregation": "bidirectional_mean",
        "filename": "overlap.npz",
        "near_depth_m": 1.2,
        "pixel_stride": 8,
        "relative_depth_tolerance": 0.03,
        "schema_version": 1,
    }
    profile_path = exported_staging / "scenes/scene_000000/overlap.npz"
    with np.load(profile_path, allow_pickle=False) as profile:
        assert set(profile.files) == {"all_depth", "near_depth", "pair_valid"}
        all_depth = profile["all_depth"]
        near_depth = profile["near_depth"]
        pair_valid = profile["pair_valid"]
    with np.load(exported_staging / "scenes/scene_000000/sequences.npz", allow_pickle=False) as sequences:
        sequence_count, max_frames = sequences["sequences"].shape
        expected_shape = (sequence_count, max_frames, max_frames)
    assert all_depth.shape == expected_shape
    assert near_depth.shape == expected_shape
    assert pair_valid.shape == expected_shape
    assert all_depth.dtype == np.float32
    assert near_depth.dtype == np.float32
    assert pair_valid.dtype == np.bool_
    assert np.isfinite(all_depth).all() and np.isfinite(near_depth).all()
    assert ((all_depth >= 0) & (all_depth <= 1)).all()
    assert ((near_depth >= 0) & (near_depth <= 1)).all()
    assert np.allclose(all_depth, all_depth.transpose(0, 2, 1))
    assert np.allclose(near_depth, near_depth.transpose(0, 2, 1))
    assert not np.diagonal(pair_valid, axis1=1, axis2=2).any()
    assert PRIVATE_TOKEN.encode() not in profile_path.read_bytes()


def test_existing_anonymous_staging_can_be_upgraded_with_overlap_profile(exported_staging: Path) -> None:
    profile_path = exported_staging / "scenes/scene_000000/overlap.npz"
    profile_path.unlink()
    metadata_path = exported_staging / "dataset.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["overlap"]
    metadata_path.write_text(json.dumps(metadata))

    report = prepare_overlap_profile(exported_staging)

    assert report["valid"]
    assert profile_path.is_file() and not profile_path.is_symlink()
    assert validate_staging(exported_staging)["valid"]
    assert all(
        PRIVATE_TOKEN.encode() not in path.read_bytes() for path in exported_staging.rglob("*") if path.is_file()
    )


def test_staging_validator_rejects_paths_names_symlinks_and_png_metadata(exported_staging: Path) -> None:
    (exported_staging / f"{PRIVATE_TOKEN}.txt").write_text(f"/private/{PRIVATE_TOKEN}")
    (exported_staging / "private_link").symlink_to(exported_staging / "dataset.json")
    rgb_path = exported_staging / "scenes/scene_000000/rgb/frame_000000.png"
    rgb = Image.open(rgb_path).copy()
    info = PngImagePlugin.PngInfo()
    info.add_text("source", PRIVATE_TOKEN)
    rgb.save(rgb_path, pnginfo=info)

    report = validate_staging(exported_staging, private_tokens=(PRIVATE_TOKEN,))

    assert not report["valid"]
    assert report["privacy"]["symlink_count"] >= 1
    assert report["privacy"]["invalid_path_count"] >= 1
    assert report["privacy"]["private_token_hit_count"] >= 1
    assert report["privacy"]["png_metadata_count"] >= 1
    assert PRIVATE_TOKEN not in json.dumps(report)


def test_staging_validator_rejects_noncontiguous_frame_ids_and_wrong_depth_dtype(exported_staging: Path) -> None:
    camera_path = exported_staging / "scenes/scene_000000/cameras.npz"
    with np.load(camera_path, allow_pickle=False) as cameras:
        arrays = {key: cameras[key].copy() for key in cameras.files}
    arrays["frame_ids"][0] = 99
    np.savez_compressed(camera_path, **arrays)
    depth_path = exported_staging / "scenes/scene_000000/depth/frame_000000.png"
    Image.fromarray(np.ones((48, 64), dtype=np.uint8)).save(depth_path)

    report = validate_staging(exported_staging)

    assert not report["valid"]
    assert not report["dataset"]["contract_valid"]


def test_staging_validator_rejects_split_file_that_points_to_another_split(exported_staging: Path) -> None:
    train_entry = (exported_staging / "splits/train.txt").read_text().splitlines()[0]
    (exported_staging / "splits/smoke.txt").write_text(train_entry + "\n")

    report = validate_staging(exported_staging)

    assert not report["valid"]
    assert report["splits"]["entry_violation_count"] == 1


def test_validate_only_cli_checks_an_existing_staging(exported_staging: Path) -> None:
    assert main(["--validate-only", "--output-root", str(exported_staging)]) == 0


def test_validate_only_cli_can_rederive_private_tokens_from_source(tmp_path: Path) -> None:
    source, trajectory, camera_yaml = _write_private_source(tmp_path, frame_count=15)
    output = tmp_path / "staging"
    export_colmap_rgbd_training(
        source_rgbd_root=source,
        trajectory_path=trajectory,
        rgb_camera_yaml=camera_yaml,
        output_root=output,
        height=48,
        width=64,
        guard_frames=1,
        split_fractions=(0.6, 0.2, 0.2),
    )

    assert (
        main(
            [
                "--validate-only",
                "--output-root",
                str(output),
                "--source-rgbd-root",
                str(source),
                "--trajectory",
                str(trajectory),
                "--rgb-camera-yaml",
                str(camera_yaml),
            ]
        )
        == 0
    )


def test_exporter_can_require_the_expected_source_frame_count(tmp_path: Path) -> None:
    source, trajectory, camera_yaml = _write_private_source(tmp_path, frame_count=15)

    with pytest.raises(ExportContractError, match="expected frame count"):
        export_colmap_rgbd_training(
            source_rgbd_root=source,
            trajectory_path=trajectory,
            rgb_camera_yaml=camera_yaml,
            output_root=tmp_path / "staging",
            height=48,
            width=64,
            expected_frame_count=1087,
            guard_frames=1,
            split_fractions=(0.6, 0.2, 0.2),
        )


def test_exporter_rejects_a_missing_pose_with_a_generic_frame_error(tmp_path: Path) -> None:
    source, trajectory, camera_yaml = _write_private_source(tmp_path, frame_count=15)
    payload = json.loads(trajectory.read_text())
    del payload["frames"][3]["camera_to_world"]
    trajectory.write_text(json.dumps(payload))

    with pytest.raises(ExportContractError, match=r"frame 3.*camera_to_world"):
        export_colmap_rgbd_training(
            source_rgbd_root=source,
            trajectory_path=trajectory,
            rgb_camera_yaml=camera_yaml,
            output_root=tmp_path / "staging",
            height=48,
            width=64,
            guard_frames=1,
            split_fractions=(0.6, 0.2, 0.2),
        )


def test_loader_returns_normalized_training_contract_deterministically(exported_staging: Path) -> None:
    first = ColmapRgbdDataset(exported_staging, split="train", min_frames=2, max_frames=4, seed=13)
    second = ColmapRgbdDataset(exported_staging, split="train", min_frames=2, max_frames=4, seed=13)

    sample = first[0]
    repeated = second[0]
    sequence_length = sample["images"].shape[0]
    assert 2 <= sequence_length <= 4
    assert sample["images"].shape == (sequence_length, 3, 48, 64)
    assert sample["depths"].shape == (sequence_length, 48, 64)
    assert sample["depth_masks"].shape == (sequence_length, 48, 64)
    assert sample["extrinsics"].shape == (sequence_length, 3, 4)
    assert sample["intrinsics"].shape == (sequence_length, 3, 3)
    assert sample["cam_points"].shape == (sequence_length, 48, 64, 3)
    assert sample["world_points"].shape == (sequence_length, 48, 64, 3)
    assert sample["images"].dtype == torch.float32
    assert sample["depths"].dtype == torch.float32
    assert sample["depth_masks"].dtype == torch.bool
    assert sample["extrinsics"].dtype == torch.float32
    assert sample["intrinsics"].dtype == torch.float32
    assert sample["frame_ids"].dtype == torch.int64
    assert 0 <= sample["images"].min() <= sample["images"].max() <= 1
    assert torch.equal(sample["frame_ids"], repeated["frame_ids"])
    expected_identity = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1)
    assert torch.allclose(sample["extrinsics"][0], expected_identity, atol=1e-5)
    valid = sample["world_points"][sample["depth_masks"]]
    assert torch.linalg.vector_norm(valid, dim=-1).mean().item() == pytest.approx(1.0, rel=1e-5)


def test_loader_collates_fixed_length_samples_with_expected_batch_shapes(exported_staging: Path) -> None:
    dataset = ColmapRgbdDataset(exported_staging, split="train", min_frames=2, max_frames=2, seed=3)

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch["images"].shape == (2, 2, 3, 48, 64)
    assert batch["depths"].shape == (2, 2, 48, 64)
    assert batch["depth_masks"].shape == (2, 2, 48, 64)
    assert batch["extrinsics"].shape == (2, 2, 3, 4)
    assert batch["intrinsics"].shape == (2, 2, 3, 3)
    assert batch["cam_points"].shape == (2, 2, 48, 64, 3)
    assert batch["world_points"].shape == (2, 2, 48, 64, 3)
    assert batch["normalization_scale_m"].shape == (2,)


def test_exported_splits_are_frame_pair_disjoint_and_chunk_local(exported_staging: Path) -> None:
    report = json.loads((exported_staging / "reports/export_validation.json").read_text())

    assert report["splits"]["frame_overlap_count"] == 0
    assert report["splits"]["pair_overlap_count"] == 0
    assert report["splits"]["chunk_boundary_violation_count"] == 0
    assert all((exported_staging / f"splits/{split}.txt").read_text().strip() for split in ("train", "val", "smoke"))


def test_loader_fails_explicitly_for_too_short_sequence(exported_staging: Path) -> None:
    dataset = ColmapRgbdDataset(exported_staging, split="val", min_frames=4, max_frames=4, seed=0)

    with pytest.raises(DataContractError, match="shorter"):
        dataset[0]


def test_loader_rejects_metadata_that_claims_a_second_depth_dilation(exported_staging: Path) -> None:
    metadata_path = exported_staging / "dataset.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["depth"]["mapped_depth_dilation_kernel"] = 5
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(DataContractError, match="mapped-depth contract"):
        ColmapRgbdDataset(exported_staging, split="train")


@pytest.mark.parametrize("missing", ["metadata", "file"])
def test_loader_rejects_missing_overlap_profile_without_random_fallback(exported_staging: Path, missing: str) -> None:
    if missing == "metadata":
        metadata_path = exported_staging / "dataset.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["overlap"]
        metadata_path.write_text(json.dumps(metadata))
    else:
        (exported_staging / "scenes/scene_000000/overlap.npz").unlink()

    with pytest.raises(DataContractError, match="overlap profile"):
        ColmapRgbdDataset(exported_staging, split="train")


def test_overlap_curriculum_target_covers_start_middle_and_end() -> None:
    assert overlap_curriculum_target(0, 5, start_target=0.8, end_target=0.2) == pytest.approx(0.8)
    assert overlap_curriculum_target(2, 5, start_target=0.8, end_target=0.2) == pytest.approx(0.5)
    assert overlap_curriculum_target(4, 5, start_target=0.8, end_target=0.2) == pytest.approx(0.2)
    assert overlap_curriculum_target(9, 5, start_target=0.8, end_target=0.2) == pytest.approx(0.2)


def test_overlap_selector_uses_target_range_then_explicit_nearest_fallback() -> None:
    scores = np.asarray([[0.0, 0.8, 0.2], [0.8, 0.0, 0.5], [0.2, 0.5, 0.0]], dtype=np.float32)
    valid = np.asarray([[False, True, True], [True, False, True], [True, True, False]], dtype=np.bool_)

    selected, score, fallback = select_overlap_frame_offsets(
        scores, valid, sample_length=2, target=0.78, tolerance=0.03, rng=np.random.default_rng(4)
    )
    nearest, nearest_score, nearest_fallback = select_overlap_frame_offsets(
        scores, valid, sample_length=2, target=0.95, tolerance=0.01, rng=np.random.default_rng(4)
    )

    assert set(selected.tolist()) == {0, 1}
    assert score == pytest.approx(0.8)
    assert not fallback
    assert set(nearest.tolist()) == {0, 1}
    assert nearest_score == pytest.approx(0.8)
    assert nearest_fallback


def test_overlap_dataset_is_reproducible_with_multiple_workers(exported_staging: Path) -> None:
    options = {
        "split": "train",
        "min_frames": 2,
        "max_frames": 3,
        "seed": 17,
        "overlap_curriculum_enabled": True,
        "overlap_start_target": 0.8,
        "overlap_end_target": 0.2,
        "overlap_target_tolerance": 0.05,
        "overlap_curriculum_epochs": 3,
    }
    first = ColmapRgbdDataset(exported_staging, **options)
    second = ColmapRgbdDataset(exported_staging, **options)
    first.set_epoch(1)
    second.set_epoch(1)

    first_batches = list(DataLoader(first, batch_size=1, shuffle=False, num_workers=2))
    second_batches = list(DataLoader(second, batch_size=1, shuffle=False, num_workers=2))

    assert len(first_batches) == len(second_batches)
    for left, right in zip(first_batches, second_batches, strict=True):
        assert torch.equal(left["frame_ids"], right["frame_ids"])
        assert torch.equal(left["sampling_overlap_score"], right["sampling_overlap_score"])
        assert torch.equal(left["sampling_overlap_target"], right["sampling_overlap_target"])
        assert torch.equal(left["sampling_overlap_fallback"], right["sampling_overlap_fallback"])


def test_overlap_curriculum_can_be_enabled_for_smoke_training_split(exported_staging: Path) -> None:
    dataset = ColmapRgbdDataset(
        exported_staging,
        split="smoke",
        min_frames=2,
        max_frames=2,
        overlap_curriculum_enabled=True,
        overlap_start_target=0.8,
        overlap_end_target=0.2,
        overlap_curriculum_epochs=3,
    )

    sample = dataset[0]

    assert set(sample) >= {
        "sampling_overlap_score",
        "sampling_overlap_target",
        "sampling_overlap_fallback",
    }
