# 公開内容マニフェスト

GitHub `MIA-laboratory/AM_video` で公開する内容の記録。

## 方針

- 公開対象 = **時系列(制約付きDP)コードのみ**。フレーム分類器のコードは非公開。
- 論文との一致は問わない（分類器非依存の時系列コードで、Inception-ResNet-v2予測で動作）。
- 学習済みモデル・集計CSV・患者データ（動画・フレーム・工程時刻）は**全て除外**。
- ライセンス: **MIT**。

## 検証済みの事実

- 本コードに **Inception-ResNet-v2(MATLAB) の Model 1 予測** を入力して構成Aを実行 →
  **overall %MAE = 1.922%（median 0.421%）**。コードが Inception-ResNet-v2 予測で
  正しく動作することを確認（CPU実行）。
- 動画長はフレーム名から算出する実装で、**患者画像フォルダなしで動作**。
- 構文チェック全4ファイルOK、EfficientNet/`work`参照・個人情報・絶対パスの残存なし。

## 公開ファイル

| ファイル | 内容 |
|---------|------|
| `python_temporal/temporal_analysis.py` | 制約付きDP・平滑化・%MAE・可視化 |
| `python_temporal/v6_temporal.py` | 構成A/B/C/D の駆動CLI |
| `python_temporal/generate_fig5.py` | 最良症例タイムライン図の生成 |
| `python_temporal/config.py` | クラス定義・DP制約・I/Oパス（環境変数で上書き可） |
| `README.md` | 使い方・入力CSV形式・実行例・ライセンス・引用 |
| `LICENSE` | MIT License |
| `.gitignore` | 患者データ・モデルの誤コミット防止 |

## 公開前の整備

- 分類器(EfficientNet)固有の記述・`work/`参照・GPU/モデル設定を全削除。
- 入力を Inception-ResNet-v2 の予測CSV に向ける構成へ（環境変数指定）。
- `get_video_durations` を画像フォルダ依存 → 予測CSV由来に改修（患者画像不要）。
- ユーザー名パス・絶対パスを除去し環境変数ベースに一般化。個人情報・トークン混入なし。

## 除外（非公開）

患者データ（動画・フレーム・`supervised_phaseTime.xlsx`）/ 学習済みモデル（`*.mat`,
`*.pth`）/ 予測CSV・okprobs / 集計CSV・図・原稿・ポスター / MATLAB分類学習コード。
