import pandas as pd


def preprocessing(train_data, test_data):
    """
    生データの前処理を行う関数。

    Parameters
    ----------
    train_data : pd.DataFrame
        学習用の生データ。
    test_data : pd.DataFrame
        テスト用の生データ

    Returns
    -------
    train_data : pd.DataFrame
        前処理済みの学習用データ。
    test_data : pd.DataFrame
        前処理済みのテスト用データ。

    Notes
    -----
    - id 列を削除
    - 特徴量名をリネーム
    """
    all_data = pd.concat([train_data, test_data])
    all_data = all_data.drop("id", axis=1)

    all_data = all_data.rename(columns={
        "Nitrogen": "N",
        "Potassium": "K",
        "Phosphorous": "P",
        "Fertilizer Name": "target",
        "Temparature": "Tem",
        "Humidity": "Hum",
        "Moisture": "Moi",
        "Crop Type": "Crop",
        "Soil Type": "Soil"
    })

    tr_df = all_data.iloc[:len(train_data)].copy()
    test_df = all_data.iloc[len(train_data):].copy()

    return tr_df, test_df