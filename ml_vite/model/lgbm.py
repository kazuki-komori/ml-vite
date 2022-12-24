import pandas as pd
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from typing import Callable

from ml_vite.util.timer import Timer


def fit_lgbm(
    X: pd.DataFrame,
    y,
    cv,
    params,
    score_func: Callable[[np.ndarray, np.ndarray], float],
  ):
  """LightGBMでfitする"""
  oof_pred = np.zeros(len(X), dtype=np.float32)
  models = []
  scores = []

  for i, (idx_train, idx_valid) in enumerate(cv):
    x_train, y_train = X[X.index.isin(idx_train)], y[idx_train]
    x_valid, y_valid = X[X.index.isin(idx_valid)], y[idx_valid]

    ds_train = lgbm.Dataset(x_train, y_train)
    ds_valid = lgbm.Dataset(x_valid, y_valid)

    with Timer(prefix=f"fit ========== Fold: {i + 1}"):
      model = lgbm.train(
        params,
        ds_train,
        valid_names=["train, valid"],
        valid_sets=[ds_train, ds_valid]
      )

      pred_i = model.predict(x_valid, num_iteration=model.best_iteration)
      oof_pred[idx_valid] = pred_i

      if score_func.__name__ == "accuracy_score":
        score = score_func(y_valid, np.round(pred_i).astype(int))
        print(f" - fold{i + 1} - :\t {score}")
      else:
        score = score_func(y_valid, pred_i)
        print(f" - fold{i + 1} - :\t {score}")

      scores.append(score)
      models.append(model)
      
  if score_func.__name__ == "accuracy_score":
    score = score_func(y, np.round(oof_pred).astype(int))
  else:
    score = score_func(y, oof_pred)
    
  print("=" * 50)
  print(f"FINISH: Whole Score: {score:.4f}")

  return oof_pred, models, score


def visualize_importance(models, feat_train_df, top_n):
  """lightGBM の model 配列の feature importance を plot する
  CVごとのブレを boxen plot として表現

  args:
      models:
          List of lightGBM models
      feat_train_df:
          学習時に使った DataFrame
  """
  feature_importance_df = pd.DataFrame()
  for i, model in enumerate(models):
    _df = pd.DataFrame()
    _df["feature_importance"] = model.feature_importance(importance_type='split')
    _df["column"] = feat_train_df.columns
    _df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, _df], 
                                        axis=0, ignore_index=True)

  order = feature_importance_df.groupby("column")\
    .sum()[["feature_importance"]]\
    .sort_values("feature_importance", ascending=False).index[:top_n]

  fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
  sns.boxenplot(data=feature_importance_df, 
                x="feature_importance", 
                y="column", 
                order=order, 
                ax=ax, 
                palette="viridis", 
                orient="h")
  ax.tick_params(axis="x", rotation=90)
  ax.set_title("Importance")
  ax.grid()
  fig.tight_layout()
  plt.show()
  return fig, ax