import pandas as pd
from ml_vite.util.timer import Timer

class AbstractBaseBlock:
  """BaseBlock 定義"""
  def __init__(self, parent_blocks = None):
    self.parent_blocks = [] if parent_blocks is None else parent_blocks

  def fit(self, input_df: pd.DataFrame, y=None):        
    return self.transform(input_df)

  def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError()


def run_blocks(input_df, blocks, y=None, test=False):
  """BaseBlock 実行関数"""
  df_out = pd.DataFrame()

  for block in blocks:
    if block.parent_blocks:
      _df = run_blocks(input_df=input_df, blocks=block.parent_blocks, y=y, test=test)
    else:
      _df = input_df

    with Timer(prefix='\t- {}'.format(str(block))):
      if not test:
        out_i = block.fit(_df, y=y)
      else:
        out_i = block.transform(_df)

    assert len(input_df) == len(out_i), block
    name = block.__class__.__name__
    df_out = pd.concat([df_out, out_i.add_suffix(f"@{name}")], axis=1)

  return df_out

def help():
  # TODO ヘルプの実装
  print(AbstractBaseBlock.__name__)
