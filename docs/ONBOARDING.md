# LLMオンボーディングサマリー

> 新任LLMエージェントが `vggt-omega`（yuki-inaho フォーク）に参加する際の初期資料です。
> 数値・コマンドはこのマシン（RTX PRO 4000 Blackwell / cu130）での検証実測に基づきます。
> 機械固有の検証記録は [`docs/SETUP_BLACKWELL.md`](./SETUP_BLACKWELL.md) を参照。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** VGGT-Omega（`vggt-omega`）。画像／動画列からの**フィードフォワード型カメラ姿勢・深度復元（Structure-from-Motion / 3D再構成）**。Oxford VGG + Meta AI の VGGT-Ω モデルを uv パッケージ化した社内フォーク。
- **最終成果物:** 入力フレーム列から、カメラ外部・内部パラメータ、深度マップ、world points（点群）を推論し、Rerun / GLB で可視化する一連のパイプライン（CLI・Gradio デモ）。
- **ビジネス背景・価値:** トマト収穫ロボット向けデータ生成基盤の3D再構成コンポーネント。VGGT-Omegaは推論結果と標準出力を提供し、下流の再構成処理とは独立して運用する。
- **現時点の進捗サマリ:** 本マシン（Blackwell / cu130）で uv 環境構築 → import 検証 → gated チェックポイント取得 → `just smoke`（4フレームGPU推論）まで**検証完了**。記録は `blackwell-cu130` ブランチ。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ライン。
- **Python は 3.10 以上 3.14 未満**（`pyproject.toml` と `uv.lock` の解決対象）。本機での検証Pythonは3.11です。
- **各リポジトリは独立した `.venv`** を `<repo>/.venv` に持つ。`uv sync --all-extras`で作成し、他repoのvenvと共有しない。
- **`numpy<2` を維持**（pyproject 制約）。
- **torch は CUDA-13 ドライバで動くビルド（cu130）**。本機ドライバは 580 / CUDA 13.0、GPU は Blackwell **sm_120**。
- **gated チェックポイント取得には承認済み `HF_TOKEN` が必須**（`facebook/VGGT-Omega`）。
- **大型成果物はコミット禁止**（`*.pt` / `checkpoints/` / `.venv/` / `outputs/` は `.gitignore` 済み）。
- **`main` への直接 push は不可**。機械固有ブランチ（例 `blackwell-cu130`）で作業し PR 経由。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| プロジェクト概要 | `README.md` | モデル概要・Quick Start・GPUメモリ表 |
| 機械別セットアップ記録 | `docs/SETUP_BLACKWELL.md` | 本機（Blackwell/cu130）の検証済み構成・確認値・メモリ実測 |
| 依存・ロック | `pyproject.toml` / `uv.lock` | 依存と extras（`demo`/`viz`/`rgbd`）、固定バージョン |
| タスクランナー | `justfile` | `sync` / `smoke` / `demo` / `viz-*` / `lint` / `test` 等の正準コマンド |
| テスト資産 | `tests/` | pytest（`gpu` / `slow` マーカー。CPU既定は `not gpu`） |
| RGB-D metric pose | `docs/RGBD_POSE_WORKFLOW.md` | RGB-D scale補正、重複chunkの初期整合、マスク点群融合。 |
| 既知課題リスト | TBD | 未整備（必要に応じて `docs/` に追加） |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- uv 環境構築（`uv sync --extra demo --extra viz --python 3.11`）と import 検証。
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
1. `uv sync --extra demo --extra viz --python 3.11` で `.venv` を作成し、`torch.cuda.is_available()==True` と device capability `(12,0)`（sm_120）を確認する。
2. `just smoke` を実行し、出力 `images=(4,3,384,688) depth=(4,384,688,1) world_points=(4,384,688,3)`（rc=0）を再現する。
3. メモリ監視（GPU/RAM を1秒サンプリング）しながらフレーム数を変え、GPUピークを `README.md` のメモリ表と突き合わせる（本機の4フレーム実測: GPU ~5.6GB / RAM ~31GB）。
4. 標準化済みRGB-Dセッションがある場合は、`uv sync --extra rgbd` の後に `docs/RGBD_POSE_WORKFLOW.md` の短い2チャンクスモークを実行する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 数値は実測コマンド出力に基づき記載し、対象ブランチ／日付を明記する。
- **TBDの扱い:** 未確認・未整備は「TBD」と明示し、断定や推測の混入を避ける。
- **レビュー/承認フロー:** 機械固有ブランチで作業 → `main` へは PR 経由でレビュー。
- **その他の運用ルール:** per-machine ブランチ命名は `<GPU世代>-cu<確認したCUDAビルド番号>`（本機 = `blackwell-cu130`）。

## 8. COLMAP 連携の境界
- 本リポジトリは VGGT-Omega の推論と予測出力だけを担う。COLMAPは依存として導入しない。
- `vggt-omega export-colmap` は推定済みの姿勢・内部パラメータを COLMAP text 形式と `predictions.npz` に**書き出すだけ**であり、COLMAP の Python API／実行ファイルを呼び出さない。
- 出力後の最適化・特徴マッチングは本リポジトリの責務外である。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `yuki-inaho/vggt-omega`（本repo）。関連: `colmap`（ONNX ALIKED/LightGlue 対応フォーク）、`tomato_*` 群。パッケージ本体は `vggt_omega/`、CLI は `vggt_omega/cli.py`、可視化は `scripts/visualize.py`。
- **代表的なコマンド:** `uv sync --extra demo --extra viz --python 3.11` / `just smoke` / `just demo` / `just viz-rrd` / `just check`。
- **依存ライブラリ（本機検証値）:** `torch 2.12.0+cu130`, `torchvision 0.27.0`, `triton 3.7.0`, `onnxruntime 1.24.3`, `gradio`, `rerun-sdk`, `viser`, `playwright`, `numpy<2`。
- **連絡先/責任者:** リポジトリオーナー `yuki-inaho`（詳細 TBD）。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理の対象です。
