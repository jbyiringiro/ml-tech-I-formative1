"""Recurrent neural forecasters -- LSTM and GRU (Task 3).

Both neural models share one implementation, :class:`NeuralForecaster`,
parameterised by the recurrent cell type. This keeps the architecture, training
loop and evaluation identical so that any performance difference is
attributable to the cell, not to incidental implementation differences.

Input representation
--------------------
The history ``x_t`` is a window of the ``lookback`` most recent (scaled)
traffic values for the area, shaped ``(lookback, 1)``. The network outputs a
single scalar -- the one-step-ahead prediction ``x_(t+1)`` -- which is then
mapped back to original traffic units with the fitted scaler.

Architecture
------------
``Input -> [recurrent layer(s)] -> Dropout -> Dense(1)``. With ``layers > 1``
the intermediate recurrent layers return full sequences and feed the next
layer. Optimisation uses Adam on the mean-squared-error loss with
early stopping on a chronological validation split.

* **LSTM** (Long Short-Term Memory) -- gated cell with an explicit cell state;
  input/forget/output gates let it retain information over long lags.
* **GRU** (Gated Recurrent Unit) -- a lighter gated cell (reset/update gates,
  no separate cell state); fewer parameters, usually faster to train.
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG
from ..preprocessing import ForecastData
from ..utils import set_global_seed, timer
from .base import ForecastResult


class NeuralForecaster:
    """Configurable recurrent forecaster (LSTM or GRU)."""

    def __init__(
        self,
        cell: str = "LSTM",
        units: int | None = None,
        layers: int | None = None,
        dropout: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
        learning_rate: float | None = None,
        verbose: int = 1,
    ) -> None:
        cell = cell.upper()
        if cell not in {"LSTM", "GRU"}:
            raise ValueError("cell must be 'LSTM' or 'GRU'")
        cfg = CONFIG.neural
        self.cell = cell
        self.name = cell
        self.units = units or cfg["units"]
        self.layers = layers or cfg["layers"]
        self.dropout = cfg["dropout"] if dropout is None else dropout
        self.epochs = epochs or cfg["epochs"]
        self.batch_size = batch_size or cfg["batch_size"]
        self.patience = patience or cfg["patience"]
        self.learning_rate = learning_rate or cfg["learning_rate"]
        self.verbose = verbose
        self.model_ = None

    # ------------------------------------------------------------------
    def build(self, input_shape: tuple[int, int]):
        """Construct and compile the Keras model for the given input shape."""
        from tensorflow import keras

        set_global_seed()
        cell_layer = keras.layers.LSTM if self.cell == "LSTM" else keras.layers.GRU

        model = keras.Sequential(name=f"{self.cell}_forecaster")
        model.add(keras.layers.Input(shape=input_shape))
        for i in range(self.layers):
            model.add(
                cell_layer(
                    self.units,
                    return_sequences=(i < self.layers - 1),
                    name=f"{self.cell.lower()}_{i + 1}",
                )
            )
        if self.dropout > 0:
            model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1, name="output"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        self.model_ = model
        return model

    # ------------------------------------------------------------------
    def fit_predict(self, data: ForecastData) -> ForecastResult:
        """Train on the train/val windows and predict the held-out test week."""
        from tensorflow import keras

        if self.model_ is None:
            self.build(data.input_shape)

        early = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )

        with timer(f"{self.cell} fit (area {data.square_id})") as t_fit:
            history = self.model_.fit(
                data.X_train,
                data.y_train,
                validation_data=(data.X_val, data.y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early],
                verbose=self.verbose,
            )

        with timer(f"{self.cell} predict (area {data.square_id})") as t_pred:
            scaled_pred = self.model_.predict(
                data.X_test, batch_size=self.batch_size, verbose=0
            ).ravel()

        # Back to original traffic units; clip negatives (traffic >= 0).
        y_pred = np.clip(data.inverse(scaled_pred), 0.0, None).astype("float32")

        return ForecastResult(
            model_name=self.cell,
            square_id=data.square_id,
            y_true=np.asarray(data.y_test_actual, dtype="float32"),
            y_pred=y_pred,
            test_index=data.test_index,
            train_time_s=float(t_fit["seconds"]),
            predict_time_s=float(t_pred["seconds"]),
            extra={
                "cell": self.cell,
                "units": self.units,
                "layers": self.layers,
                "dropout": self.dropout,
                "lookback": data.lookback,
                "batch_size": self.batch_size,
                "epochs_run": len(history.history["loss"]),
                "best_val_loss": float(min(history.history["val_loss"])),
                "n_params": int(self.model_.count_params()),
                "history": {k: [float(v) for v in vals]
                            for k, vals in history.history.items()},
            },
        )

    def summary(self) -> str:  # pragma: no cover - convenience
        if self.model_ is None:
            return f"{self.cell} model not yet built."
        lines: list[str] = []
        self.model_.summary(print_fn=lines.append)
        return "\n".join(lines)


def build_lstm(**kwargs) -> NeuralForecaster:
    """Convenience constructor for an LSTM forecaster."""
    return NeuralForecaster(cell="LSTM", **kwargs)


def build_gru(**kwargs) -> NeuralForecaster:
    """Convenience constructor for a GRU forecaster."""
    return NeuralForecaster(cell="GRU", **kwargs)
