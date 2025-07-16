import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineering(train_data, test_data, weight=0.4):
    """
    特徴量エンジニアリングを行う関数

    Parameters
    ----------
    train_data : pd.DataFrame
        前処理済みの学習用データ
    test_data : pd.DataFrame
        前処理済みのテスト用データ
    weight : float
        originalデータのweight

    Returns
    -------
    tr_df : pd.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pd.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - 数値変数をカテゴリ化する
    - originalデータを追加
    """
    original_data = pd.read_csv(
        "../artifacts/features/original_data.csv"
    )
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, original_data, test_data], ignore_index=True
    )
    all_data["const"] = 1

    # === 1) 数値変数をカテゴリー化 ===
    num_df = all_data.select_dtypes(include=[np.number])
    num_df = num_df.astype("category")

    # === 2) カテゴリー変数をlabel encoding ===
    cat_cols = ["Soil", "Crop"]
    cat_le_df = pd.DataFrame(index=all_data.index)

    for c in cat_cols:
        le = LabelEncoder()
        cat_le_df[c] = le.fit_transform(all_data[c])

    # === dfを結合 ===
    df_feat = pd.concat([num_df, cat_le_df], axis=1)

    # === データを分割 ===
    tr_len = len(train_data)
    org_len = len(original_data)

    tr_df = df_feat.iloc[:tr_len + org_len].copy()
    test_df = df_feat.iloc[tr_len + org_len:].copy()

    # === target と weight を追加 ===
    target = pd.concat([
        train_data["target"],
        original_data["target"]
    ], axis=0).reset_index(drop=True)

    weights = np.concatenate([
        np.ones(tr_len),
        np.full(org_len, weight)
    ])

    tr_df["target"] = target
    tr_df["weight"] = weights

    return tr_df, test_df