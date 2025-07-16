import optuna
import math
from src.models.lgb.lgbm_cv_trainer import LGBMCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    cat_cols=None,
    n_jobs=20
):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        訓練データ。
    n_splits : int, default 5
        CV分割数。
    early_stopping_rounds : int, default 200
        EarlyStoppingのラウンド数。
    cat_cols : list, default None
        カテゴリ変数名のリスト。
    n_jobs: int, default 20
        LGBM並列数。

    Returns
    -------
    objective : function
        optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "learning_rate": 0.1,
            "max_depth": trial.suggest_int("max_depth", 7, 8),
            "num_leaves": trial.suggest_int("num_leaves", 500, 800),
            "min_child_samples": trial.suggest_int("min_child_samples",
                                                   5000, 15000),
            "min_split_gain": trial.suggest_float("min_split_gain",
                                                  1e-5, 10, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction",
                                                    0.25, 0.45),
            "bagging_fraction": trial.suggest_float("bagging_fraction",
                                                    0.65, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
            "lambda_l1": trial.suggest_float("lambda_l1",
                                             1e-5, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2",
                                             1e-5, 10.0, log=True),
            "n_jobs": n_jobs,
        }

        min_required_depth = int(math.log2(params["num_leaves"])) + 1
        params["max_depth"] = max(params["max_depth"], min_required_depth)

        trainer = LGBMCVTrainer(
            params=params,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            cat_cols=cat_cols
        )

        trainer.fit_one_fold(tr_df, fold=0)

        best_iteration = trainer.fold_models[0].model.best_iteration
        trial.set_user_attr("best_iteration", best_iteration)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1, study_name="lgbm_study",
    storage=None, initial_params: dict = None, sampler=None
):
    """
    Optunaによるハイパーパラメータ探索を実行する関数。

    Parameters
    ----------
    objective : function
        Optunaの目的関数。
    n_trials : int, default 50
        試行回数。
    n_jobs : int, default 1
        並列実行数。
    study_name : str or None, default "lgbm_study"
        StudyName。
    storage : str or None, default None
        保存先URL。
    initial_params : dict or None, default None
        初期の試行パラメータ。
    sampler : optuna.samplers.BaseSampler or None, default TPESampler
        使用するSampler。

    Returns
    -------
    study : optuna.Study
        探索結果のStudyオブジェクト。
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler or optuna.samplers.TPESampler()
    )

    if initial_params is not None:
        study.enqueue_trial(initial_params)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    return study