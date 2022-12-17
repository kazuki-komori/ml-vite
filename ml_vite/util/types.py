from typing import Union, Iterable, Optional
from sklearn.model_selection import BaseCrossValidator

t_cv = Optional[Union[int, Iterable, BaseCrossValidator]]