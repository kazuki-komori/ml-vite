import numpy as np
from typing import Callable, List

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


class NNModel:
  """Neaural Networkモデルの構築"""
  def __init__(
    self,
    input_size: int,
    hidden_sizes: List,
    output_size: int,
    activation: str = "relu",
    output_activation: str = "relu",
    add_BN: bool = True
    ) -> None:
    """NN module を"""
    self.input = Input(shape=(input_size, ), name="input")
    self.output_size = Dense(output_size, name="output")
    self.add_BN = add_BN

    self.hidden_layers = []
    for idx, hidden_size in enumerate(hidden_sizes):
      if (idx+1) == len(self.hidden_layers):
        self.hidden_layers.append(Dense(hidden_size, output_activation, name=f"hidden_{idx+1}"))
        continue
      self.hidden_layers.append(Dense(hidden_size, activation, name=f"hidden_{idx+1}"))
  
  def build(self) -> Model:
    inputs = self.input
    for idx, hidden_layer in enumerate(self.hidden_layers):
      if idx == 0:
        x = hidden_layer(inputs)
        continue

      x = BatchNormalization()(x)
      x = hidden_layer(x)
    outputs = self.output_size(x)
    return Model(inputs, outputs)


def fit_nn(
  X,
  y,
  cv, 
  metrics: List,
  score_func: Callable[[np.ndarray, np.ndarray], float],
  NNModel: Model, 
  NN_MODEL_PARAMS: dict,
  n_epoch: int = 500, 
  loss: str = "mse",
  optimizer = "adam",
  verbose: int = -1,
  eary_stopping_rounds: int = 50,
  ):
  """NNの学習"""
  oof_pred = np.zeros(len(X), dtype=np.float32)
  models = []
  scores = []

  for i, (idx_train, idx_valid) in enumerate(cv):
    x_train, y_train = X[X.index.isin(idx_train)], y[idx_train]
    x_valid, y_valid = X[X.index.isin(idx_valid)], y[idx_valid]

    model = NNModel(**NN_MODEL_PARAMS)
    model = model.build()
    model.compile(
      optimizer,
      loss,
      metrics
    )
    history = model.fit(
          x_train,
          y_train,
          epochs=n_epoch,
          verbose=verbose,
          validation_data=(x_valid, y_valid),
          callbacks=[EarlyStopping(patience=eary_stopping_rounds)]
        )

    plot_history(history)

    oof = model.predict(x_valid)
    oof_pred[idx_valid] = oof.flatten()
    models.append(model)
    
    print("-"*50)
    if score_func.__name__ == "accuracy_score":
      print(f"score {i+1}:\t {score_func(y_valid, np.round(oof).astype(int))}")
    else:
      print(f"score {i+1}:\t {score_func(y_valid, oof)}")

  print("*"*50)
  if score_func.__name__ == "accuracy_score":
    score = score_func(y, np.round(oof_pred).astype(int))
  else:
    score = score_func(y, oof_pred)
  print(f"score {i+1}:\t {score}")

  return models, oof_pred, score


def plot_history(hist):
  # 損失値(Loss)の遷移のプロット
  plt.plot(hist.history['loss'],label="train set")
  plt.plot(hist.history['val_loss'],label="test set")
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.show()