import pandas as pd
import numpy as np
from typing import Callable

import xgboost as xgb

from ml_forge.util.timer import Timer


def fit_xgb(
    X: pd.DataFrame, 
    y, 
    cv, 
    params, 
    score_func: Callable[[np.ndarray, np.ndarray], float],
    early_stopping_rounds: int = 20,
  ):
  oof_pred = np.zeros(len(X), dtype=np.float32)
  models = []
  scores = []

  for i, (idx_train, idx_valid) in enumerate(cv):
    x_train, y_train = X[X.index.isin(idx_train)], y[idx_train]
    x_valid, y_valid = X[X.index.isin(idx_valid)], y[idx_valid]

    ds_train = xgb.DMatrix(x_train, y_train)
    ds_valid = xgb.DMatrix(x_valid, y_valid)

    with Timer(prefix=f"fit ========== Fold: {i + 1}"):
      model = xgb.train(
        params,
        dtrain=ds_train,
        evals=[(ds_train, "train"), (ds_valid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        num_boost_round=500000,
        verbose_eval=200
      )

      pred_i = model.predict(ds_valid, ntree_limit=model.best_ntree_limit)
      oof_pred[idx_valid] = pred_i

      if score_func.__name__ == "accuracy_score":
        print(f" - fold{i + 1} - :\t {score_func(y_valid, np.round(oof_pred).astype(int))}")
      else:
        print(f" - fold{i + 1} - :\t {score_func(y_valid, oof_pred)}")
      print(f" - fold{i + 1} - {score:.4f}")

      scores.append(score)
      models.append(model)
    
  if score_func.__name__ == "accuracy_score":
    score = score_func(y, np.round(oof_pred).astype(int))
  else:
    score = score_func(y, oof_pred)
    
  print("=" * 50)
  print(f"FINISH: Whole Score: {score:.4f}")

  return oof_pred, models, score