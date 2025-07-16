import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineering(train_data, test_data):
    """
    特徴量エンジニアリングを行う関数

    Parameters
    ----------
    train_data : pd.DataFrame
        前処理済みの学習用データ
    test_data : pd.DataFrame
        前処理済みのテスト用データ

    Returns
    -------
    tr_df : pd.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pd.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - 数値変数をbin分割してカテゴリ化する
    - 交互作用を追加
    - 比率の特徴を追加
    """
    all_data = pd.concat([train_data, test_data])
    num_df = all_data.select_dtypes(include=[np.number])
    num_cols = num_df.columns

    # === 1) 数値変数をbin分割化
    bin_df = pd.DataFrame(index=all_data.index)
    num_cols = all_data.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        bin_df[c] = pd.cut(all_data[c], bins=50, labels=False)

    # ==== 2) カテゴリ変数のみ
    cat_df = all_data[["Soil", "Crop"]]
    cat_cols = cat_df.columns
    cat_df_le = pd.DataFrame(index=all_data.index)
    for c in cat_cols:
        le = LabelEncoder()
        cat_df_le[f"{c}_le"] = le.fit_transform(cat_df[c])

    # === 3) カテゴリと bin の交互作用
    cat_bin_df = pd.concat([bin_df, cat_df], axis=1)
    inter_df = pd.DataFrame()
    cols = cat_bin_df.columns

    # 低次交互作用
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            inter_df[f"{c1}_{c2}_inter"] = (
                cat_bin_df[c1].astype(str) + "_" +
                cat_bin_df[c2].astype(str)
            )

    # 高次交互作用
    inter_df["N_P_K"] = (
        cat_bin_df["N"].astype(str) + "_" +
        cat_bin_df["P"].astype(str) + "_" + 
        cat_bin_df["K"].astype(str)
    )
    inter_df["Temp_Hum_Moi"] = (
        cat_bin_df["Tem"].astype(str) + "_" +
        cat_bin_df["Hum"].astype(str) + "_" +
        cat_bin_df["Moi"].astype(str)
    )

    # === 4) Label Encoding
    inter_le_df = pd.DataFrame(index=all_data.index)

    for c in inter_df.columns:
        le = LabelEncoder()
        inter_le_df[f"{c}"] = le.fit_transform(inter_df[c])

    # === 5) NPK比率
    npk_df = pd.DataFrame(index=all_data.index)
    npk_total = all_data["N"] + all_data["P"] + all_data["K"] + 1e-6
    npk_df["N_ratio"] = all_data["N"] / npk_total
    npk_df["P_ratio"] = all_data["P"] / npk_total
    npk_df["K_ratio"] = all_data["K"] / npk_total

    npk_df["NPK_total"] = npk_total
    npk_df["P/K"] = all_data["P"] / (all_data["K"] + 1e-6)
    npk_df["N/K"] = all_data["N"] / (all_data["K"] + 1e-6)
    npk_df["P/N"] = all_data["P"] / (all_data["N"] + 1e-6)

    npk_df["PN_over_PK"] = npk_df["P/N"] / npk_df["P/K"]
    npk_df["PN_over_NK"] = npk_df["P/N"] / npk_df["N/K"]
    npk_df["NK_over_PK"] = npk_df["N/K"] / npk_df["P/K"]

    finite_vals = (
        npk_df["NK_over_PK"][np.isfinite(npk_df["NK_over_PK"])]
    )
    max_val = finite_vals.max()
    npk_df["NK_over_PK"] = (
        npk_df["NK_over_PK"].replace([np.inf, -np.inf], max_val)
    )

    # === 6) 追加の農業関連特徴量
    various_ratio_df = pd.DataFrame(index=all_data.index)

    # 環境ストレス指標
    various_ratio_df["(Hum-Moi)/Tem"] = (
        (all_data["Hum"] - all_data["Moi"]) /
        (all_data["Tem"] + 1e-6)
    )
    various_ratio_df["(Tem-Hum)/Moi"] = (
        (all_data["Tem"] - all_data["Hum"]) /
        (all_data["Moi"] + 1e-6)
    )
    various_ratio_df["(Tem-Moi)/Hum"] = (
        (all_data["Tem"] - all_data["Moi"]) /
        (all_data["Hum"] + 1e-6)
    )

    # 効率性指標
    various_ratio_df["Hum/Tem"] = (
        all_data["Hum"] / (all_data["Tem"] + 1e-6)
    )
    various_ratio_df["Moi/Hum"] = (
        all_data["Moi"] / (all_data["Hum"] + 1e-6)
    )
    various_ratio_df["Moi/Tem"] = (
        all_data["Moi"] / (all_data["Tem"] + 1e-6)
    )

    # dfを結合
    df_feat = pd.concat([
        num_df,
        cat_df_le,
        inter_le_df,
        npk_df,
        various_ratio_df
    ], axis=1)

    tr_df = df_feat.iloc[:len(train_data)]
    test_df = df_feat.iloc[len(train_data):].copy()
    tr_df["target"] = train_data["target"]

    return tr_df, test_df