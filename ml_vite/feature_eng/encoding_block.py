from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_vite.util.types import t_cv
from ml_vite.util.error import CustomErrorMsg
from ml_vite.feature_eng.base_block import AbstractBaseBlock

class TargetEncodingBlock(AbstractBaseBlock):
  """Target Encoding"""
  def __init__(self, target_col, agg: List, cv: t_cv, **kwrgs) -> None:
    super().__init__(**kwrgs)
    self.target = target_col
    self.agg = agg
    self.cv = cv

  def fit(self, df_input: pd.DataFrame, y):
    self.dics_ = []
    if y not in df_input.columns:
      raise CustomErrorMsg(f"df_input に目的変数 {y} がありません")

    df_out = pd.DataFrame(index=df_input.index)

    for idx_feature, idx_valid in self.cv:
      # train と valid を作成
      df_feature = df_input[df_input.index.isin(idx_feature)]
      df_valid = df_input[df_input.index.isin(idx_valid)]
      _dic = df_feature.groupby(self.target)[y].agg(self.agg).to_dict()
      self.dics_.append(_dic)
      # agg で回す
      for agg in self.agg:
        df_out.loc[idx_valid, f"target={self.target}_agg_func={agg}"] = df_valid[self.target].map(_dic[agg])
    return df_out

  def transform(self, df_input: pd.DataFrame):
    df_out = pd.DataFrame()
    for agg in self.agg:
      _df = pd.DataFrame()
      for i, _dic in enumerate(self.dics_):
        _df[i] = df_input[self.target].map(_dic[agg])
      df_out[f"target={self.target}_agg_func={agg}"] = _df.mean(axis=1)
    return df_out


class OneHotEncodingBlock(AbstractBaseBlock):
    """ダミー変数の作成"""
    def __init__(self, target_col: str, drop_first: bool, **kwrgs) -> None:
        super().__init__(**kwrgs)
        self.target_col = target_col
        self.drop_first = drop_first
    
    def fit(self, df_input: pd.DataFrame, y=None):
        df_train_out = pd.get_dummies(df_input[self.target_col], drop_first=self.drop_first).add_prefix(f"{self.target_col}_")
        self.train_column_ = df_train_out.columns
        return df_train_out

    def transform(self, df_input: pd.DataFrame):
        return pd.get_dummies(df_input[self.target_col], drop_first=self.drop_first).add_prefix(f"{self.target_col}_").reindex(columns=self.train_column_, fill_value=0)


class CountEncodingBlock(AbstractBaseBlock):
  """CountEncoding"""
  def __init__(self, target_col: str, **kwrgs) -> None:
    super().__init__(**kwrgs)
    self.target_col = target_col

  def fit(self, df_input: pd.DataFrame, y=None):
    self.vc_ = df_input[self.target_col].value_counts()
    return self.transform(df_input)

  def transform(self, df_input: pd.DataFrame):
    return pd.DataFrame(df_input[self.target_col].map(self.vc_))


class LabelEncodingBlock(AbstractBaseBlock):
  """LabelEncoding"""
  def __init__(self, target_col: str) -> None:
    super().__init__()
    self.le = LabelEncoder()
    self.target_col = target_col

  def fit(self, df_input: pd.DataFrame, y=None):
    self.le.fit(df_input[self.target_col])
    return self.transform(df_input)

  def transform(self, df_input: pd.DataFrame):
    return pd.DataFrame({
        self.target_col: self.le.fit_transform(df_input[self.target_col])
    })