# 公開内容マニフェスト

GitHub `MIA-laboratory/AM_video` で公開する内容の記録。

## 方針

- 公開対象 = **方法論（制約付きDPによる時系列再構成）の実装のみ**。
- 分類器・OK/NGモデルは利用者が持ち込む設計（MATLAB `.mat` / Python のいずれも可）。
  論文に準拠するのは**方法論**であり、特定のモデル・データではない。
- 学習済みモデル・集計CSV・患者データ（動画・フレーム・工程時刻）は**全て同梱しない**。
- ライセンス: **MIT**。

## 検証済みの事実

- 本コードに **Inception-ResNet-v2(MATLAB) の Model 1 予測** を入力して構成A–Dが
  正常に動作することを確認（CPU実行）。%MAE の正本値は論文 Table 6：
  構成A（OK/NGなし）overall 2.00%（median 0.50%）、構成B（OK/NG @ θ=0.5）
  overall 1.92%（median 0.33%）。完全再現には Table 6 と同一の予測CSV・設定
  （平滑化窓・候補サブサンプリング）が必要。
- 動画長はフレーム名から算出する実装で、**患者画像フォルダなしで動作**。
- 全Pythonファイルの構文チェックOK、EfficientNet/`work`参照・個人情報・絶対パスの残存なし。

## 公開ファイル

| ファイル | 内容 |
|---------|------|
| `python_temporal/temporal_analysis.py` | 制約付きDP・平滑化・%MAE・可視化 |
| `python_temporal/v6_temporal.py` | 構成A/B/C/D の単一実行CLI |
| `python_temporal/run_v6.py` | 構成×OK/NG閾値を一括実行する駆動スクリプト |
| `python_temporal/baselines.py` | 系列ベースライン比較（論文 Table 10） |
| `python_temporal/generate_fig5.py` | 最良症例タイムライン図の生成（論文 Figure 7） |
| `python_temporal/config.py` | クラス定義・DP制約・I/Oパス（環境変数で上書き可） |
| `okng/export_okprobs_matlab.m` | 自分のMATLAB `.mat` OK/NGモデルからOK確率CSVを出力するテンプレート |
| `okng/export_okprobs_python.py` | 自分のPython OK/NGモデルからOK確率CSVを出力するテンプレート |
| `README.md` | 使い方・入力形式・OK/NGモデル持ち込み手順・ライセンス・引用 |
| `LICENSE` | MIT License |
| `.gitignore` | 患者データ・モデルの誤コミット防止 |

OK確率の入力は **CSV / parquet 両対応**（`path`, `ok_prob` 列）。MATLAB/Python どちらの
OK/NGモデルからでも出力でき、構成B/Dに適用可能。構成A/CはOK/NG不要で予測CSVのみで動作。

## 公開前の整備

- 分類器(EfficientNet)固有の記述・`work/`参照・GPU/モデル設定を全削除。
- 入力を Inception-ResNet-v2 の予測CSV に向ける構成へ（環境変数指定）。
- `get_video_durations` を画像フォルダ依存 → 予測CSV由来に改修（患者画像不要）。
- ユーザー名パス・絶対パスを除去し環境変数ベースに一般化。個人情報・トークン混入なし。

## 除外（非公開）

患者データ（動画・フレーム・`supervised_phaseTime.xlsx`）/ 学習済みモデル（`*.mat`,
`*.pth`）/ 予測CSV・okprobs / 集計CSV・図・原稿・ポスター / MATLAB分類学習コード。
