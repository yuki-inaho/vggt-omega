# LLMオンボーディングサマリー（`gtx1070` ブランチ）

> 新任LLMエージェントが `vggt-omega`（yuki-inaho フォーク）に参加する際の初期資料です。
> 本書は **`gtx1070` ブランチ専用**で、数値・コマンドは本マシン
> （**GeForce GTX 1070 / Pascal sm_61 / torch cu126 ビルド**）での検証実測に基づきます。
> このブランチは `blackwell-cu130` から派生し、差分は実質「torch の CUDA ビルド
> （cu130 → cu126）＋ AMP dtype の 1 行修正」のみ。プロジェクト全体像・制約・運用は
> 共通です。Blackwell マシン（RTX PRO 4000 / cu130）の機械固有記録は
> [`docs/SETUP_BLACKWELL.md`](./SETUP_BLACKWELL.md) を参照（姉妹マシンの参考）。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** VGGT-Omega（`vggt-omega`）。画像／動画列からの**フィードフォワード型カメラ姿勢・深度復元（Structure-from-Motion / 3D再構成）**。Oxford VGG + Meta AI の VGGT-Ω モデルを uv パッケージ化した社内フォーク。
- **最終成果物:** 入力フレーム列から、カメラ外部・内部パラメータ、深度マップ、world points（点群）を推論し、Rerun / GLB で可視化する一連のパイプライン（CLI・Gradio デモ）。
- **ビジネス背景・価値:** トマト収穫ロボット向けデータ生成基盤の3D再構成コンポーネント。COLMAP フォーク（ONNX ALIKED/LightGlue）や GLUEMAP-VGGT 再構成（例: `TVA_NYX650_*_colmap` データセット）と連携する。
- **現時点の進捗サマリ:** 本マシン（GTX 1070 / cu126）で uv 環境構築 → GPU カーネル検証 → gated チェックポイント取得 → `just smoke`（4フレームGPU推論）→ `just viz-rrd`（6フレーム可視化）→ `just check` / `just test-gpu` まで**検証完了**。記録は `gtx1070` ブランチ（本書 §8）。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ライン。
- **Python は 3.11 以上**（ロック済み `onnxruntime` が cp311+ 専用。既定の 3.10 では `uv sync` が失敗）。
- **各リポジトリは独立した `.venv`** を `<repo>/.venv` に持つ。常に `UV_PROJECT_ENVIRONMENT="$PWD/.venv"` で固定（他repoの venv と共有しない）。
- **`numpy<2` を維持**（pyproject 制約）。
- **torch は GPU 世代に対応した CUDA ビルドを使う。** 本機 GPU は **GTX 1070 / Pascal sm_61**。**CUDA 13.0 は Maxwell/Pascal/Volta を削除済み**なので、既定 PyPI の `+cu130` ビルドでは `no kernel image is available` で動かない。本ブランチは **cu126 ビルド**（`pyproject.toml` の `[tool.uv.sources]` + `[[tool.uv.index]]` で固定）を使う。Blackwell（sm_120）機は cu130。詳細は §8。
- **AMP dtype は native bf16 のみで判定する**（§8 の修正）。Pascal は `is_bf16_supported()` がエミュレーションで True を返すが、ハード bf16 は無く遅い。
- **gated チェックポイント取得には承認済み `HF_TOKEN` が必須**（`facebook/VGGT-Omega`）。
- **大型成果物はコミット禁止**（`*.pt` / `checkpoints/` / `.venv/` / `outputs/` / `*.rrd` は `.gitignore` 済み）。
- **`main` への直接 push は不可**。機械固有ブランチ（本機 = `gtx1070`）で作業し PR 経由。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| プロジェクト概要 | `README.md` | モデル概要・Quick Start・GPUメモリ表 |
| 本機の検証記録 | 本書 §8 | GTX 1070 / cu126 の検証済み構成・確認値・メモリ実測（統合済み） |
| 姉妹機の検証記録 | `docs/SETUP_BLACKWELL.md` | Blackwell / cu130 機の検証記録（参考） |
| 依存・ロック | `pyproject.toml` / `uv.lock` | 依存と extras（`demo`/`viz`/`export`）、cu126 インデックス、固定バージョン |
| タスクランナー | `justfile` | `sync` / `smoke` / `demo` / `viz-*` / `lint` / `test` 等の正準コマンド |
| テスト資産 | `tests/` | pytest（`gpu` / `slow` マーカー。CPU既定は `not gpu`） |
| 既知課題リスト | TBD | 未整備（必要に応じて `docs/` に追加） |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- uv 環境構築（`uv sync --extra demo --extra viz --python 3.11`）と GPU カーネル検証。
- 推論実行（`just smoke` / `just demo`）と Rerun 可視化（`just viz-rrd` / `viz-screenshot`）。
- lint / format / typecheck / test（`just check`）。
- gated チェックポイント取得（`HF_TOKEN` 経由）とドキュメント整備。

### 任せないタスク
- モデル本体の学習・重み改変、アーキテクチャ変更。
- HuggingFace の gated アクセス権限付与・承認操作。
- `main` への直接 push、大型成果物（`*.pt` 等）のコミット。
- 本番データ・トークン等の秘匿情報の外部送信。

## 5. インタラクション方針
- **回答スタイル:** 日本語、見出し＋箇条書き。確認値・実測値は表で提示。
- **回答手順:** 前提（環境確認）→ 検証（実測コマンド）→ 提案／実行 の順。
- **禁止事項・注意:** 未確定事項を断定しない（不明は「TBD」と明示）。`HF_TOKEN` の値をログ・コミット・出力に残さない。
- **秘匿情報の扱い:** `HF_TOKEN` は環境変数経由のみで使用。`~/.bashrc` 等の定義値は出力しない。

## 6. 試行タスク（オンボーディング演習）
1. `uv sync --extra demo --extra viz --python 3.11` で `.venv` を作成し、`torch.cuda.is_available()==True` と device capability `(6,1)`（sm_61）を確認する。`torch.cuda.is_bf16_supported(including_emulation=False)==False`（→ fp16 経路）も確認する。
2. `just smoke` を実行し、出力 `images=(4, 3, 384, 688) depth=(4, 384, 688, 1) world_points=(4, 384, 688, 3)`（rc=0）を再現する。
3. メモリ監視（GPU/RAM を1秒サンプリング）しながらフレーム数を変え、GPUピークを `README.md` のメモリ表と突き合わせる（本機の4フレーム実測: GPU使用ピーク ~6.66GB（うち~1GBはデスクトップ常駐）/ プログラム実体 ~5.6GB）。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 数値は実測コマンド出力に基づき記載し、対象ブランチ／日付を明記する。
- **TBDの扱い:** 未確認・未整備は「TBD」と明示し、断定や推測の混入を避ける。
- **レビュー/承認フロー:** 機械固有ブランチで作業 → `main` へは PR 経由でレビュー。
- **per-machine ブランチ命名:** 原則 `<GPU世代>-cu<確認したCUDAビルド番号>`（例 `blackwell-cu130`）。本ブランチは依頼により GPU 名そのままの **`gtx1070`**（実体は Pascal / cu126）。

## 8. 機械固有の検証記録（GeForce GTX 1070 / cu126）
本マシンの再現性検証済みセットアップ記録。`blackwell-cu130` から派生し、ランタイム差分は
PyTorch の CUDA ビルド（cu130 → **cu126**）と AMP dtype の 1 行修正のみ。

### なぜ別ビルドが要るか
GTX 1070 は **Pascal**、compute capability **sm_61**。**NVIDIA CUDA 13.0 が
Maxwell/Pascal/Volta サポートを削除**したため、既定 PyPI の `torch`（`+cu130`、Blackwell 機で
使用）は本 GPU 向けカーネルを一切持たず `no kernel image is available for execution on the
device` で落ちる。対策は **CUDA 12.6 ビルド**の torch/torchvision を入れること。cu126 wheel は
旧アーキを含み、GTX 1070 は CUDA のマイナー版前方互換（`sm_60` cubin → `sm_61` で実行）により
ネイティブ動作する。新しい CUDA-13 ドライバ（580.x）は cu126 バイナリを問題なく実行する。

### 確認済みハードウェア / ドライバ
| 項目 | 確認値 |
|---|---|
| GPU | NVIDIA GeForce GTX 1070, 8192 MiB (8 GB) |
| Compute capability | **sm_61**（`torch.cuda.get_device_capability` → `(6, 1)`） |
| ドライバ | 580.159.03 |
| ドライバ CUDA 版 | **13.0**（後方互換で cu126 ビルドを実行） |

### 環境（uv）
- **Python 3.11.14**（3.10 不可：ロック済み `onnxruntime` が cp311+ 専用）。
- cu126 ソースは `pyproject.toml` で固定：
  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cu126" }]
  torchvision = [{ index = "pytorch-cu126" }]

  [[tool.uv.index]]
  name = "pytorch-cu126"
  url = "https://download.pytorch.org/whl/cu126"
  explicit = true
  ```
- venv 作成（`<repo>/.venv`）：
  ```bash
  UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --extra demo --extra viz --python 3.11
  ```
- 確認済み主要バージョン：`torch 2.12.1+cu126`, `torchvision 0.27.1+cu126`,
  `triton 3.7.1`, `onnxruntime 1.26.0`。`torch.cuda.is_available()` → `True`、
  device `NVIDIA GeForce GTX 1070`。
- `torch.cuda.get_arch_list()` → `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`。
  sm_61 は無いが `sm_60` cubin が GTX 1070 で動く（前方互換）。

### AMP 精度（Pascal 固有の修正）
`torch.cuda.is_bf16_supported()` は本 GPU で**ソフトウェアエミュレーション**により **True** を返すが、
Pascal にハード bf16 は無い。実測でエミュ bf16 matmul は fp16 比 ~2倍遅い（2048² GEMM で
6.55ms vs 3.35ms）。そこで `vggt_omega/models/vggt_omega.py` は AMP dtype を
`is_bf16_supported(including_emulation=False)`（ネイティブ判定）で選ぶよう変更。Pascal では
**fp16**（upstream が非 bf16 ハード向けに用意していた経路）に、Ampere/Blackwell では従来通り
bf16 に解決されるため、Blackwell 機の挙動は不変。

### チェックポイント（gated）
- `facebook/VGGT-Omega` は HF で gated。承認済み `HF_TOKEN` をエクスポートしてから取得：
  ```bash
  HF_TOKEN=... uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import hf_hub_download as d; \
    print(d('facebook/VGGT-Omega','vggt_omega_1b_512.pt', local_dir='checkpoints'))"
  ```
- `checkpoints/vggt_omega_1b_512.pt` → 4.58 GB。

### 検証済み実行 + メモリプロファイル
- スモークテスト（fp16 経路）：
  ```bash
  just smoke   # forest_road.mp4, 4 frames
  ```
  結果：`smoke ok: images=(4, 3, 384, 688) depth=(4, 384, 688, 1) world_points=(4, 384, 688, 3)`（rc=0）
  — Blackwell 機と同一出力。
- 正当性サニティ（4フレーム）：depth 100% 有限・平均 ≈ 0.94、intrinsics 妥当
  （fx≈413, fy≈416, cx=344, cy=192）、pose_enc 全有限、world_points 100% 有限。
- ピーク GPU メモリ実測（1秒サンプリング、**~1.0 GB のデスクトップ/Xorg 常駐を含む**）：
  | 実行 | GPU使用ピーク | GPU空き最小 | 実時間 |
  |---|---|---|---|
  | `just smoke`（4フレーム） | **6,658 MiB** | 1,441 MiB | ~61 s |
  | `just viz-rrd`（6フレーム） | **7,109 MiB** | 991 MiB | ~75 s |
  - デスクトップ ~1GB を引いたプログラム実体は 4 フレームで ≈ 5.6 GB。Blackwell 記録
    （5,716 MiB）および upstream README 表とほぼ一致。
  - 実時間は uv 起動 + 4.58GB チェックポイントロードを含む（純推論はその一部）。Pascal の fp16 は
    tensor core 非搭載のため新世代 GPU より遅い。

### 8 GB での実用上限
デスクトップで ~1GB 消費されるため実質 ~7GB。README メモリ表（1f≈6.0GB, 10f≈6.7GB,
50f≈9.7GB）より、GTX 1070 は 512px で概ね **~10 フレームまで**が快適で、50 フレームでは OOM する。
長尺は `mode="max_size"` か小さい `--image-resolution`、または `--num-frames` 削減で対応。

### 品質ゲート
`just check`（format-check + lint-check + ty typecheck + CPUテスト）グリーン、
`just test-gpu`（GPUテスト1件）も本機でパス。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `yuki-inaho/vggt-omega`（本repo）。関連: `colmap`（ONNX ALIKED/LightGlue 対応フォーク）、`tomato_*` 群。パッケージ本体は `vggt_omega/`、CLI は `vggt_omega/cli.py`、可視化は `scripts/visualize.py`。
- **代表的なコマンド:** `uv sync --extra demo --extra viz --python 3.11` / `just smoke` / `just demo` / `just viz-rrd` / `just check` / `just test-gpu`。
- **依存ライブラリ（本機検証値）:** `torch 2.12.1+cu126`, `torchvision 0.27.1+cu126`, `triton 3.7.1`, `onnxruntime 1.26.0`, `gradio`, `rerun-sdk`, `viser`, `playwright`, `numpy<2`。
- **連絡先/責任者:** リポジトリオーナー `yuki-inaho`（詳細 TBD）。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理の対象です。
