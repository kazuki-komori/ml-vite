import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm
from typing import List, Dict

class LGBMImputer:
  def __init__(
      self,
      feature_cols: List,
      target_cols: List,
      early_stopping_rounds: int = 20
    ) -> None:
    self.feature_cols = feature_cols
    self.target_cols = target_cols
    self.early_stopping_rounds = early_stopping_rounds
    self.models: Dict[str, lgbm.Booster] = {}

  def fit(self, df_input: pd.DataFrame):
    for target_col in tqdm(self.target_cols, desc="train lgbm models..."):
      # 欠損の行をデータセットとする
      train_ds, valid_ds = self._create_ds(df_input, target_col)
      params = self._feature_type_checker(train_ds.label)
      # 予測
      model = lgbm.train(
        {**params, "verbosity": -1},
        num_boost_round=500000,
        train_set=train_ds,
        callbacks=[
            lgbm.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=-1)
        ],
        valid_sets=[valid_ds]
      )
      self.models[target_col] = model
      
    return self.transform(df_input)


  def transform(self, df_input: pd.DataFrame):
    df_input_cp = df_input.copy(deep=True)

    for target_col in tqdm(self.target_cols, desc="predict from lgbm models..."):
      if df_input[target_col].isnull().sum() > 0:
        # 欠損の行をデータセットとする
        _, valid_ds = self._create_ds(df_input, target_col)
        df_input_cp.loc[valid_ds.data.index, target_col] = self.models[target_col].predict(valid_ds.data, num_iteration=self.models[target_col].best_iteration)
    return df_input_cp


  def _create_ds(self, df_input: pd.DataFrame, target_col: str):
    # 欠損の行をデータセットとする / 欠損補完するカラムはテストデータとする
    idx_miss = df_input[df_input[target_col].isnull()].index
    if len(idx_miss) != 0:
      train_ds = lgbm.Dataset(
        df_input[~df_input.index.isin(idx_miss)][self.feature_cols],
        df_input[~df_input.index.isin(idx_miss)][target_col],
      )
      valid_ds = lgbm.Dataset(
        df_input[df_input.index.isin(idx_miss)][self.feature_cols],
        df_input[df_input.index.isin(idx_miss)][target_col],
      )
    else:
      # 訓練データに欠損値がないとき
      df_train, df_valid = train_test_split(df_input, test_size=.2, random_state=712)
      train_ds = lgbm.Dataset(
        df_input[df_input.index.isin(df_train.index)][self.feature_cols],
        df_input[df_input.index.isin(df_train.index)][target_col],
      )
      valid_ds = lgbm.Dataset(
        df_input[df_input.index.isin(df_valid.index)][self.feature_cols],
        df_input[df_input.index.isin(df_valid.index)][target_col],
      )
    return train_ds, valid_ds

  def _feature_type_checker(self, series: pd.Series) -> dict:
    # 自動で型を判定してパラメータを返す
    if pd.api.types.is_numeric_dtype(series):
      # 連続変数のとき
      return {
        'objective': 'regression'
      }
    else:
      # カテゴリカル変数のとき
      nuni = series.dropna().nunique()
      if nuni == 2:
        return {
          'objective': 'binary'
        }
      elif nuni > 2:
        return {
          'objective': 'multiclass',
          'num_class': nuni + 1
        }