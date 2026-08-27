# 標準化済みロボットRGB-DのVGGT pose workflow

この文書は、標準化済みロボットRGB-Dセッションを入力として、VGGT-OmegaによるRGB推論、RGB-Dによるmetric scale補正、重複chunkの初期整合、点群融合、次段階のpose graph最適化までを定義します。

## 1. 前提と方針

- 入力セッションは`rgb/`、`mapped_depth_dense/`、`point_clouds_left_third_or_stem_foreground_voxel_0025m/`を含みます。
- `mapped_depth_dense/`はRGB FoVへ整列済みのuint16深度画像で、単位はmmです。
- 高速推論はネイティブPyTorchのBF16 autocastを使用します。RTX 4090、640x480、6フレームではORTより高速・低メモリでした。
- VGGTの深度とposeは相対scaleです。RGB-D深度との比からchunkごとに一つのmetric scaleを推定し、depthとpose並進へ適用します。
- 現行のchunk整合は共有フレームのVGGT poseを用いる**初期SE(3)連鎖**です。ICP、loop closure、異なるセッションの座標統合は行いません。

## 2. セットアップ

```bash
cd /workspace/inaho_repos/vggt-omega
uv sync --extra rgbd
```

GPUとチェックポイントを確認します。

```bash
nvidia-smi
test -f checkpoints/vggt_omega_1b_512.pt
```

標準化済みデータの例です。

```text
/home/kasm-user/Desktop/data/2026-07-17_tmt4-04_standardized/<session-id>/
  rgb/
  mapped_depth_dense/
  stem_masks/
  point_clouds_left_third_or_stem_foreground_voxel_0025m/
```

生成物は`outputs/`に置き、Gitへ追加しません。

## 3. 単一chunkのRGB推論とmetric pose

連続6フレームを640x480へ縮小し、BF16で推論します。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_vggt_rgbd_pose_workflow.py \
  --session-dir /home/kasm-user/Desktop/data/2026-07-17_tmt4-04_standardized/8417f984-3f2d-44d9-a0a6-fb18b3736dfc \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir outputs/vggt_rgbd/example_6f_640x480
```

処理内容:

1. RGBをBICUBICで640x480へ縮小し、対応する`mapped_depth_dense`をNEARESTで同じサイズへ縮小する。
2. VGGTのネイティブPyTorch BF16経路で`pose_enc`と相対depthを推論する。
3. metric depthが0.10〜5.00m、予測depthが正の画素について`metric_depth / predicted_depth`を算出する。
4. 比の5〜95パーセンタイルを除外し、中央値を`rgbd_scale`とする。
5. `scaled_depth_m = rgbd_scale * predicted_depth`、`scaled_translation = rgbd_scale * predicted_translation`を保存する。

成果物:

- `summary.json`: 入力フレーム、scale、有効画素数
- `vggt_rgbd_pose_results.npz`: 相対／metric depth、intrinsic、camera-from-world pose、camera centre
- `camera_trajectory_rgbd_scaled.png`: 3D軌跡
- `camera_translation_by_frame.png`: X/Y/Z並進の時系列

単一scaleの適用は、VGGT depthとpose並進が同じsimilarity scaleに従うという前提です。ここで得る座標系はchunk内ローカル座標系です。

## 4. 重複chunkによる全セッションの初期整合

6フレームchunk、stride 3では隣接chunkが3フレーム重なります。

```text
chunk 0: frames 0..5
chunk 1: frames 3..8
chunk 2: frames 6..11
```

全セッションの初期整合と、マスク済み点群の5mm voxel融合を実行します。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_vggt_rgbd_chunk_alignment.py \
  --session-dir /home/kasm-user/Desktop/data/2026-07-17_tmt4-04_standardized/8417f984-3f2d-44d9-a0a6-fb18b3736dfc \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir outputs/vggt_rgbd_pose_graph_initial/8417f984-3f2d-44d9-a0a6-fb18b3736dfc \
  --chunk-size 6 --stride 3 --fusion-voxel-size-m 0.005
```

各chunkではSection 3と同じscale補正を行います。隣接chunkの共有フレームposeから`T_target_from_source`を推定し、先頭chunkをglobal原点として順に連鎖します。各フレームのマスク済みcamera-frame PLYをglobal座標へ変換し、voxel融合します。

成果物:

- `aligned_masked_clouds/*_aligned.ply`: global座標へ変換したフレーム別点群
- `fused_masked_vggt_initial_pose.ply`: 5mm voxel融合点群
- `vggt_initial_pose_alignment.npz`: frame pose、chunk pose、フレーム名
- `summary.json`: chunk scale、共有フレームedge、点群数、境界残差

短い実データsmokeは2chunk・9フレームに限定します。

```bash
VGGT_RGBD_SMOKE_SESSION=/path/to/standardized/session \
VGGT_RGBD_SMOKE_CHECKPOINT=/path/to/vggt_omega_1b_512.pt \
uv run --extra rgbd pytest -m rgbd_smoke -q tests/test_rgbd_chunk_alignment_smoke.py
```

## 5. 結果確認

必ず`summary.json`で以下を確認します。

- `chunk_count`、`unique_frame_count`: 期待フレーム数と一致すること
- `frame_observation_counts`: 重複フレームが複数観測されること
- `edge_residual_summary`: 並進／回転残差のp50、p95、worst
- `input_point_count > fused_point_count > 0`: 融合が成立したこと

PLYとNPZを再読込し、有限座標、同次行`[0, 0, 0, 1]`、フレーム数を検証します。既存の旧成果物に`edge_residual_summary`が無い場合は、現行スクリプトで再実行して出力形式を揃えます。

## 6. 次段階: pose graph最適化

初期整合は隣接変換の連鎖なので、長いセッションではドリフトします。次段階は以下です。

1. `edge_residual_summary`のp95/worstが大きい境界を抽出する。
2. 共有フレームのRGB-D点群をRANSACで粗整合し、point-to-plane ICPで`T_target_from_source`を精密化する。
3. 非隣接でも視野が再訪するchunkを検出し、loop-closure edgeを追加する。
4. chunkをノード、RANSAC/ICP変換を重み付きedgeとするSE(3) pose graphを最小二乗で最適化する。
5. 最適化済みchunk poseを各フレーム／点群へ再適用し、残差・点群の重なり・trajectoryの連続性を再評価する。

セッション間の座標系を統合するには、既知のロボットbase pose、共有ターゲット、またはセッション横断の信頼できるloop closureが別途必要です。
