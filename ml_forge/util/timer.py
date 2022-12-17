from time import time

class Timer:
  def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):
    if prefix:
      format_str = str(prefix) + sep + format_str
    if suffix:
      format_str = format_str + sep + str(suffix)
    self.format_str = format_str
    self.logger = logger
    self.start = None
    self.end = None

  @property
  def duration(self):
    if self.end is None:
      return 0
    return self.end - self.start

  def __enter__(self):
    self.start = time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end = time()
    out_str = self.format_str.format(self.duration)
    if self.logger:
      self.logger.info(out_str)
    else:
      print(out_str)
