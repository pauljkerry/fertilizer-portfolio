import numpy as np
import pandas as pd
import joblib


def create_sub(test_proba, path="../output/sub_vn.csv"):
    """
    Kaggleの提出用のフォーマットに整形する関数。

    Parameters
    ----------
    test_proba : np.ndarray
        各ラベルについての予測値の配列。
    path : str
        保存先のpath
    """
    label_encoder = joblib.load("../artifacts/label_encoder.pkl")
    top3 = np.argsort(test_proba, axis=1)[:, ::-1][:, :3]
    top3_labels = np.array(
        [label_encoder.inverse_transform(row) for row in top3]
    )
    preds = [" ".join(row) for row in top3_labels]

    sub_df = pd.DataFrame({
        "id": np.arange(750000, 750000 + len(test_proba)),
        "Fertilizer Name": preds
    })
    sub_df.to_csv(path, index=False)
    print(f"Saved model successfully to {path}!")