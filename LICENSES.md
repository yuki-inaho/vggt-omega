# License map

This repository is a mixed-license distribution. The license is determined by
the file or path, not by the repository name as a whole.

## Upstream VGGT-Omega and derivatives: FAIR Noncommercial Research License

The upstream boundary is Git commit
`39a0cb8af88554f15ddcb5354cd52bde588fa014` (`Initial commit`). Every file
present in that commit, and later modifications derived from those files,
remain subject to the [FAIR Noncommercial Research License](./LICENSE).

This scope includes, in particular:

- `LICENSE`
- `README.md`
- `demo_gradio.py`
- `visual_util.py`
- `pyproject.toml`
- `requirements.txt` and `requirements_demo.txt`
- `vggt_omega/__init__.py`
- `vggt_omega/models/**`
- `vggt_omega/utils/**`
- `vggt_omega/inference.py`, `vggt_omega/pipeline.py`, and
  `vggt_omega/preprocess.py`, which were extracted from or directly derived
  from the upstream demo and utility implementation
- upstream model weights, outputs produced from those Research Materials, and
  other material covered by the FAIR license terms

Later changes to a FAIR-licensed file do not change the license of that file or
make the combined derivative commercially usable.

## Separable fork additions: Apache License 2.0

Subject to each contributor having the right to license their contribution,
the original, separable fork additions in the following paths are offered
under the [Apache License 2.0](./LICENSE-APACHE-2.0):

- `configs/**`
- `scripts/**`
- `tests/**`
- `ERAYZER_INTEGRATION_RESULTS.md`
- `evaluate_vggt_rgbd_burst_boundaries.py`
- `export_vggt_boundary_window_poses.py`
- `run_vggt_rgbd_chunk_alignment.py`
- `run_vggt_rgbd_pose_chain.py`
- `run_vggt_rgbd_pose_workflow.py`
- `scan_vggt_macro_segments.py`
- `justfile`
- `vggt_omega/cli.py`
- `vggt_omega/colmap_export.py`
- `vggt_omega/visualize.py`
- `vggt_omega/training/**`, except for the separately licensed AMUSE files
  listed below

Copyright in these additions remains with their respective contributors. A
file-specific notice overrides this path map when one is present.

## Third-party code

`vggt_omega/training/optim/amuse.py` is vendored third-party code. Its source,
provenance, and Apache-2.0 license are recorded in
`third_party/amuse/UPSTREAM.md` and `third_party/amuse/LICENSE`.

Dependencies, generated lock-file entries, examples, model weights, datasets,
and other third-party assets retain their own licenses and terms.

## Combined use

The Apache-2.0 grant applies only to the separable fork additions listed above.
It does not relicense Meta's VGGT-Omega Research Materials or remove their
noncommercial-use restriction. Using or distributing this repository as a
combined package requires compliance with every applicable license, including
the FAIR Noncommercial Research License.
