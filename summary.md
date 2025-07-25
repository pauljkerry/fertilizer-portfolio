## 目次
1. [コンペ期間中 → 終了後に行ったこと](#1-コンペ期間中--終了後に行ったこと)
2. [コンペ全体を通しての気づきと学び](#2-コンペ全体を通しての気づきと学び)
3. [うまくいかなかったこと](#3-うまくいかなかったこと)
4. [今後の課題](#4-今後の課題)

## 1. コンペ期間中 → 終了後に行ったこと

**提出スコア**: `0.352`  
**改善後スコア**: `0.365`

### 特徴量エンジニアリングの改善
- コンペ期間中の特徴量ファイル：`fe_gbdt_v5.py`  
- コンペ終了後の特徴量ファイル：`fe_gbdt_v4.py`

#### 主な変更点

- カテゴリ変数の交互作用を削除（モデルが自動で学習可能なため）
- SHAP 値が低かった比率系特徴量を削除
- オリジナルデータを追加

### XGBoost の設定見直し
- `enable_categorical=True`を有効にし、スコアが`0.335 → 0.355`に改善

## 2. コンペ全体を通しての気づきと学び

1. `optuna`は探索範囲が広いとき、`n_startup_trials`が少ないと局所解にはまりやすい
2. 特徴量エンジニアリングを施したデータは保存しておくと再利用しやすい
3. 長時間実験（例：Ensemble）は中間結果を保存すべき（Windows自動再起動で一度消失）
4. 複数ファイルだった訓練スクリプトを`cv_trainer.py`に統合
5. 特徴量ファイル名は`fe_gbdt_vX`のようにバージョン管理し、内容は Notion に記録
6. `optuna`の`study`をSQLite → Docker + PostgreSQLに移行し、複数PCで共有可能に

## 3. うまくいかなかったこと
1. 異なるKFOLDの分割シードで学習して平均をとるアンサンブルでは、訓練データのスコアは改善したがテストデータのスコアはほとんど改善しなかった
    - 仮説：アンサンブル数が少なかった?

2. full train によるスコア改善も限定的
    - 仮説：イテレーション数やその他パラメータの最適化が不十分だった?

## 4. 今後の課題
1. `NN`の実装
2. `CatBoost`でスコアが出なかった原因の究明