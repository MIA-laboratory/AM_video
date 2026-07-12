# 公開内容マニフェスト

GitHub `MIA-laboratory/AM_video` で公開する内容の記録。

## 方針

- 公開対象 = **方法論（非単調 隠れセミマルコフモデル HSMM による時系列再構成）の実装のみ**。
- 分類器・OK/NGモデルは利用者が持ち込む設計（分類器は PyTorch、OK/NGは Python `.pth` /
  MATLAB `.mat` のいずれも可）。論文に準拠するのは**方法論**であり、特定のモデル・データではない。
- 学習済みモデル・集計CSV・患者データ（動画・フレーム・工程時刻）は**全て同梱しない**。
- ライセンス: **MIT**。

## 手法（論文 Rev3 に準拠）

- 平滑化した毎フレーム事後確率を放射確率とし、(i) 教師フェーズ列から学習した
  **phase→phase 遷移行列**（一方向固定順を課さない・後戻り許容・自己遷移禁止・
  未観測遷移は floor ε=10⁻³）、(ii) フェーズ毎の**最小持続**（semi-Markov）、
  (iii) 過剰分割を抑える**スイッチペナルティ λ** を用いた segmental DP で復号する。
- 遷移行列は**各 fold の学習16例のみ**から推定し、評価対象のテスト例は使用しない
  （リーク回避）。
- 復号フェーズ列から各フェーズの初到達時刻を取り出し、教師の遷移時刻と %MAE で比較する。
- 従来の「固定順・制約付きDP」を置き換えた非単調モデルであり、20例中3例に見られる
  工程の再出現（例：ナイダス剥離中の流入血管確保への復帰）を表現できる。

## 検証済みの事実

- 本コードに **Inception-ResNet-v2（PyTorch）の Model 1 予測** を入力して構成A–Dが
  正常に動作することを確認（CPU実行）。%MAE の正本値は論文 Table 6：
  構成A（OK/NGなし）overall **3.27%**（median 0.78%, n=69）、構成B（Model1+OK/NG @ θ=0.5）
  3.86%、構成C（Model3）3.95%、構成D（Model3+OK/NG）3.95%。20例では推論時OK/NGゲートは
  %MAE を改善しないため、構成A（ゲートなし）を最終的な運用点とする。
- フェーズ別 %MAE（構成A）：開頭 0.53%、流入血管確保 0.95%、ナイダス剥離 9.32%、
  閉頭 1.12%、手術終了 0.40%。
- ベースライン比較（論文 Table 10, Model-1 入力）：提案 HSMM 3.27%（初到達を採点するため
  n=69）に対し、変化点検出 15.0%、無順序 HMM/Viterbi 22.1%、argmax+平滑化 32.7%、
  毎フレーム argmax 46.2%（ベースラインは棄却しないため GT に存在する全境界 n=91 で採点）。
- 完全再現には Table 6 と同一の予測CSV・設定（平滑化窓90秒・候補サブサンプリング・
  λ・最小持続）が必要。
- 動画長はフレーム名から算出する実装で、**患者画像フォルダなしで動作**。
- 全Pythonファイルの構文チェックOK、`work`参照・個人情報・絶対パスの残存なし。

## 公開ファイル

| ファイル | 内容 |
|---------|------|
| `python_temporal/temporal_analysis.py` | 非単調HSMMデコーダ・遷移行列学習・平滑化・%MAE・可視化 |
| `python_temporal/v6_temporal.py` | 構成A/B/C/D の単一実行CLI |
| `python_temporal/run_v6.py` | 構成×OK/NG閾値を一括実行する駆動スクリプト |
| `python_temporal/baselines.py` | 時系列モデルのベースライン比較（論文 Table 10） |
| `python_temporal/generate_fig5.py` | 最良症例タイムライン図の生成（論文 Figure 7 = Case XII） |
| `python_temporal/config.py` | クラス定義・HSMMパラメータ・交差検証fold・I/Oパス（環境変数で上書き可） |
| `okng/export_okprobs_python.py` | 自分のPython OK/NGモデルからOK確率CSVを出力するテンプレート |
| `okng/export_okprobs_matlab.m` | 自分のMATLAB `.mat` OK/NGモデルからOK確率CSVを出力するテンプレート |
| `README.md` | 使い方・入力形式・結果・OK/NGモデル持ち込み手順・ライセンス・引用 |
| `LICENSE` | MIT License |
| `.gitignore` | 患者データ・モデルの誤コミット防止 |

OK確率の入力は **CSV / parquet 両対応**（`path`, `ok_prob` 列）。Python/MATLAB どちらの
OK/NGモデルからでも出力でき、構成B/Dに適用可能。構成A/CはOK/NG不要で予測CSVのみで動作。

## 除外（非公開）

患者データ（動画・フレーム・工程時刻xlsx）/ 学習済みモデル（`*.pth`, `*.mat`）/
予測CSV・okprobs / 集計CSV・図・原稿・ポスター / 分類器の学習コード。
