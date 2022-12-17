import datetime
import pandas as pd


def save_sub(
  df_sub: pd.DataFrame,
  nb_num: str,
  score: float,
  save_path: str
  ) -> None:
  """submission の作成"""
  now = datetime.datetime.now()
  time = now.strftime('%m_%d_%H_%M')

  df_sub.to_csv(f"{save_path}/{nb_num}_sub_{time}_{round(score, 4)}.csv", index=False)
