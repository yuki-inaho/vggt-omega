# E-RayZer-style VGGT-Omega fine-tuning results

## Experiment contract

- The public pretrained VGGT-Omega checkpoint was the immutable base model.
- The previous near-depth best head was the initial trainable state.
- Baseline and integrated runs used the same anonymous train/validation splits,
  seed (`42`), image size (`384x512`), AMUSE learning rates, and 2,896 optimizer
  steps.
- Only camera and depth heads were trainable. The backbone and confidence head
  remained frozen.
- The integrated run added near-depth overlap curriculum and relative pairwise
  camera supervision. The validation relative-pose coefficient was `0.1`.
- Checkpoint ranking used `val/objective`; lower is better. Every Top-K metric
  below was recomputed over the complete 128-sequence validation split.

## Full-run comparison

The original baseline objective did not include pairwise pose. Therefore the
table reports both the legacy camera/depth objective and the pairwise-aware
objective. Comparing the raw stored objectives directly would be misleading.

| Metric | Near-depth baseline | Integrated best (epoch 2) | Delta | Outcome |
| --- | ---: | ---: | ---: | --- |
| Legacy camera/depth objective | 0.174091899 | 0.174447802 | +0.000355903 (+0.204%) | worse |
| Pairwise-aware objective | 0.304700387 | 0.303640032 | -0.001060355 (-0.348%) | better |
| Camera loss | 0.025654432 | 0.025683877 | +0.000029445 | worse |
| Camera translation L1 | 0.023422358 | 0.023479200 | +0.000056842 | worse |
| Camera rotation L1 | 0.002024818 | 0.002014761 | -0.000010057 | better |
| Camera FoV L1 | 0.000414513 | 0.000379831 | -0.000034682 | better |
| Normalized depth L1 (`<1.2 m`) | 0.045819737 | 0.046028418 | +0.000208680 | worse |
| All-depth MAE (m) | 0.050353641 | 0.050924515 | +0.000570875 | worse |
| All-depth RMSE (m) | 0.092570695 | 0.093320063 | +0.000749368 | worse |
| Near-depth MAE (m, `<1.2 m`) | 0.046220597 | 0.046448549 | +0.000227952 | worse |
| Near-depth RMSE (m, `<1.2 m`) | 0.084685119 | 0.084770811 | +0.000085692 | worse |
| Near-depth AbsRel (`<1.2 m`) | 0.059395590 | 0.059628026 | +0.000232435 | worse |
| Pairwise pose loss | 1.306084883 | 1.291922279 | -0.014162604 | better |
| Pairwise rotation (degrees) | 0.923201074 | 0.925593222 | +0.002392148 | worse |
| Translation direction (degrees) | 72.111402810 | 71.297997549 | -0.813405260 | better |
| Translation magnitude L1 | 0.031390600 | 0.031382845 | -0.000007755 | better |
| RPA@5 | 0.000000000 | 0.000000000 | 0.000000000 | unchanged |
| RPA@15 | 0.003906250 | 0.005208333 | +0.001302083 | better |
| RPA@30 | 0.020833333 | 0.032552084 | +0.011718750 | better |

The integrated full run completed four epochs and 2,896 optimizer steps in
1,472.45 seconds (24 minutes 32 seconds), versus 1,497.26 seconds for the
baseline. Peak allocated CUDA memory was 7.235 GiB versus 7.237 GiB. These
small timing and memory differences are operationally unchanged.

The best integrated checkpoint is
`best_epoch_000002_77468befcd03.pt`, SHA-256
`57a8cd180a37e5400f281a726423eee04ae8891525d3c2e648e645b16838b72d`.
Its initial near-depth head SHA-256 was
`a1e4114691d0f271d2dbfab5a357dff4a28ab5b9aea1a8ae5aad6115c8faaa29`.

## Photometric smoke ablation

The soft-renderer photometric run was compared against pairwise-only with the
same three optimizer steps and input sampling. It added a validation
photometric L1 of `0.137793` with visibility coverage `0.106806`. Runtime was
12.19 seconds versus 11.81 seconds (+3.26%); peak CUDA memory was effectively
unchanged. Its pairwise-aware objective excluding the explicitly added
`0.01 * photometric` term did not improve relative to the pairwise-only smoke.
Consequently, the phase gate did not justify an expensive photometric full run.

## Interpretation and next experiment

Near-depth overlap scheduling plus pairwise supervision improved translation
direction, translation magnitude, RPA@15, RPA@30, and the pairwise-aware
objective. It slightly worsened absolute translation and depth. This is a
measured trade-off, not an across-the-board improvement. The available full
run measures overlap scheduling and pairwise supervision as one combined
change, so it cannot attribute the gain to either component alone. The smoke
ablation does show that adding the current photometric term on top of that
combination did not improve the non-photometric objective.

The next one-variable experiment should retain overlap sampling and reduce the
final relative-pose validation/training coefficient from `0.1` to `0.05`.
This directly tests whether the direction/RPA gains can be retained while
reducing the depth and absolute-translation regressions. Do not add the
photometric term in that experiment.

## Reproduction

```bash
uv run python scripts/train_colmap_rgbd.py \
  data=colmap_rgbd_overlap \
  model=omega_1b_512_near_head \
  loss=erayzer_pairwise_near \
  trainer=finetune \
  trainer.epochs=4 \
  trainer.max_train_steps=2896 \
  trainer.early_stopping.enabled=true \
  trainer.early_stopping.patience=4 \
  optimizer.muon_lr=1.0e-5 \
  optimizer.aux_lr=1.0e-6
```

Generated datasets, weights, checkpoints, TensorBoard events, and run outputs
are intentionally excluded from Git.
