# vggt-omega Blackwell RGB-D オンボーディング

**作成日**: 2026-09-06

**対象**: 新しいLLMエージェント、開発メンバー

**プロジェクト**: vggt-omega (`yuki-inaho/vggt-omega`)
**目的**: `blackwell-develop`上のCOLMAP RGB-D、camera/depth/correspondence学習と可視化を、private artifactを公開せず引き継ぐ。

## 1. 新しいエージェントの作業順序

1. `README.md`、`LICENSES.md`、本書を読む。
2. `git status --short --branch`で`blackwell-develop`と既存差分を確認する。
3. `git fetch origin blackwell-develop`後、remoteとのahead/behindを確認する。
4. `pixi install`を実行し、CPU testとGPU情報確認を行う。
5. 使用するdata config、model config、checkpoint SHAを確認する。
6. 1～3 stepのsmokeを別run directoryで実施する。
7. full runはsmoke、strict reload、validation契約が通った場合だけ開始する。
8. checkpointのmonitor名を確認し、Top‑1を選ぶ。
9. 推論・動画・NPZを検査する。
10. `checkpoints/`、`outputs/`、`temp/`、private dataをGitへ追加しない。

## 2. プロジェクト概要

vggt-omegaはVGGTを基盤にcamera、depth、point、correspondence等を扱うmulti-view geometryモデルである。本ブランチではCOLMAP RGB-D loader、Hydra学習管理、AMUSE optimizer、Top-K checkpoint、TensorBoard、Pixel-Perfect depth refiner、2D flow/correspondence headを扱う。

主要コンポーネント:

- `vggt_omega/models/`: aggregatorと各head
- `vggt_omega/training/`: runner、loss、optimizer、checkpoint管理
- `configs/training/`: Hydra data/model/trainer/checkpoint設定
- `scripts/train_colmap_rgbd.py`: 学習entrypoint
- `scripts/reconstruct_rgbd_video.py`: RGB-D再構成・動画出力
- `tests/`: loader、loss、checkpoint、flow、inference回帰test
- `third_party/`: licenseを保持した外部実装

## 3. 現在の状態

### 完了済み

| 分類 | 状態 | 内容 |
|---|---|---|
| COLMAP RGB-D | 完了 | 640×480、mm→m、pose/intrinsics、sequence splitを実装 |
| 学習管理 | 完了 | Hydra、TensorBoard、AMUSE、Top-K保存 |
| depth | 完了 | dense headとPixel-Perfect refinerを学習可能 |
| flow | 完了 | camera/depth幾何flowと学習residual/correspondence出力 |
| checkpoint | 完了 | wrapped checkpointのresume/readを修正 |
| Git | 完了 | `blackwell-develop`が`origin/blackwell-develop`を追跡 |

### 最新の保存対象モデル

最新FULL joint flow v4では、validation `correspondence_epe_px`をmonitorし、Top‑1はepoch 0 / step 0、16.6685 pxである。epoch 1/2より良いため選ばれているが、「追加学習で収束した」とは表現しない。モデルartifactは公開repoではなくprivate temporary storageで管理する。

### 重要な制約

- flowは画像上の2D pixel displacementであり、3D scene flowではない。
- staged RGB-Dは匿名化された単一sceneであり、raw/private sourceではない。
- absolute flow checkpointとresidual flow checkpointの意味を混同しない。
- `last.pt`をbestとみなさない。
- 可視化改善だけで定量性能改善を主張しない。

## 4. 前提条件とセットアップ

```bash
git branch --show-current
git log --oneline -1
pixi --version
pixi install
pixi run verify-gpu
```

CPU test:

```bash
CUDA_VISIBLE_DEVICES='' pixi run python -m pytest -q
```

`pyproject.toml`と`uv.lock`も存在するが、このブランチのBlackwell再現では`pixi.lock`を優先する。複数package managerの環境を混ぜない。

private dataはコマンドlineのHydra overrideまたはshell変数から渡す。例:

```bash
export VGGT_RGBD_ROOT=/path/to/private/colmap_rgbd
pixi run train-rgbd data=colmap_rgbd_640x480_fixed4 data.root="$VGGT_RGBD_ROOT"
```

値をtracked config、README、commit messageへ書かない。

## 5. 学習前確認

```bash
nvidia-smi
df -h .
git status --short
git diff --check
```

次を確認する。

- RGB/depth/maskの件数、shape、dtype、finite
- split内sequenceが重複しない
- poseが期待するw2c/c2w規約
- base checkpointとresume checkpointのSHA-256
- monitor metric、mode、Top-K数
- 新規run directoryであり、既存成果物を上書きしない

## 6. smoke、full、可視化

利用可能task:

```bash
pixi run train-rgbd --help
```

Hydra overrideでtrain/validation batch数を制限し、最初にbounded smokeを行う。smokeでは次を必須確認する。

- forward/backward/optimizer stepが1回以上成功
- loss、gradient、predictionがfinite
- checkpoint save/loadがstrictに成功
- GPU peakと空き容量が記録される

full run後の確認順序:

1. progress/run summaryがcompleteか確認
2. TensorBoard eventのscalar tagとstep数を検査
3. leaderboardのTop‑Kを再計算
4. Top‑1をstrict reloadしてvalidation再計算
5. RGB-D、flow、combined動画を生成
6. MP4の解像度、fps、frame数を`ffprobe`で検査
7. NPZのshapeとfiniteを全件検査

## 7. 成果物と作業記録

公開Gitに置くもの:

- source code、config、test
- license、sanitized onboarding

公開Gitに置かないもの:

- model weights、checkpoint、optimizer state
- RGB-D payload、latents、NPZ、動画
- TensorBoard、Hydra run directory
- session transcript、private workdoc、環境ファイル

private artifactの復元手順とSHAは、別管理の`temporally_storage`にあるREADME、MANIFEST、ONBOARDINGを参照する。

## 8. トラブルシューティング

### CUDA OOM

他process、batch、sequence length、resolution、activation checkpointを確認する。allocator fragmentationには`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`を検討する。

### resume後にflowの意味が変わる

checkpoint metadataでabsolute/residual表現を確認する。古いcheckpointを新しいresidual headへ無条件に読み込まない。

### validation値と動画の印象が違う

評価mask、pixel grid、metric scale、frame samplingを照合する。動画をmetricの代用にしない。

### dataset rootが見つからない

private rootを明示指定する。公開repo内へのfallbackやデータコピーを作らない。

## 9. 次のステップ

1. branch、remote、dirty stateの確認
2. dataset/checkpoint SHAのlocal照合
3. CPU testとbounded smoke
4. best checkpointのstrict再評価
5. depth/camera/flowを同一splitで比較
6. 代表動画とNPZの検品
7. staged diffのsecret、path、large file scan

## 10. 完了チェックリスト

- [ ] README、LICENSES、本書を読んだ
- [ ] `blackwell-develop`とremoteの同期を確認した
- [ ] Pixi環境とCPU testを確認した
- [ ] data/checkpointの契約とSHAを確認した
- [ ] smoke成功後にfullを開始した
- [ ] monitor、Top‑K、best/lastを区別した
- [ ] flowの2D/residual意味を確認した
- [ ] 動画と定量metricの両方を検査した
- [ ] private artifactをstageしていない
- [ ] `git diff --cached`とremote divergenceを確認した

## 11. 更新履歴

- 2026-09-06 UTC: Blackwell RGB-D、Pixel-Perfect depth、FULL joint flow成果に合わせて初版作成。
