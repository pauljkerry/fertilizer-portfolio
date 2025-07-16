import numpy as np


def map_k(y_true, y_preds, k=3):
    """
    map@kのスコアを算出する関数

    Parameters
    ----------
    y_true : np.ndarray
        正解ラベル。
    y_preds : np.ndarray
        予測したラベル。

    Returns
    -------
    map@3 : float
        map@3のスコア

    Notes
    -----
    - y_preds : shape = (n_samples, n_classes)
    """
    top_k = np.argsort(y_preds, axis=1)[:, ::-1][:, :k]

    def apk(actual, predicted):
        for i in range(k):
            if predicted[i] == actual:
                return 1.0 / (i + 1)
        return 0.0

    return np.mean([apk(a, p) for a, p in zip(y_true, top_k)])