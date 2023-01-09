import seaborn as sns
from tqdm import tqdm

def load_theme(theme="crest"):
  # load default theme
  tqdm.pandas()
  sns.set_style('whitegrid')
  sns.set_palette(theme)
  import japanize_matplotlib

