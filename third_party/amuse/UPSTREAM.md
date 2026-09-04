# AMUSE upstream provenance

- Project: AMUSE: Anytime Muon with Stable Gradient Evaluation
- Upstream repository: https://github.com/kjeiun/amuse
- Upstream source: `src/optim/AMUSE.py`
- Upstream commit: `48922743b32f33f919ab54edde3dbad0d0ce2dc7`
- Upstream Git blob SHA-1: `144361bf100d0a3a07172fb007a6fb27ff58f046`
- Vendored SHA-256: `84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f`
- Vendored path: `vggt_omega/training/optim/amuse.py`
- License: Apache License 2.0 (`third_party/amuse/LICENSE`)
- License SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- Local modifications: none

The upstream repository is not an installable Python package and has no release tags at the pinned revision. Only the
optimizer source required by this project is vendored; the upstream `requirements.txt` and experiment code are not
included.

To verify that the vendored bytes still match the pinned upstream Git blob:

```bash
git hash-object vggt_omega/training/optim/amuse.py
```

The command must print `144361bf100d0a3a07172fb007a6fb27ff58f046`.
