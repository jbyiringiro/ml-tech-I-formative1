"""Visualisation helpers for Task 3 (forecasting results & diagnostics).

Kept separate from :mod:`eda` (Task 2 plots) so each task's figures live in a
single, easily-located place.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .models.base import ForecastResult

sns.set_theme(style="whitegrid", context="notebook")

_MODEL_COLOURS = {"SARIMA": "darkorange", "LSTM": "royalblue", "GRU": "seagreen"}


def plot_forecasts_for_area(area_label: str, square_id: int,
                            results: list[ForecastResult]):
    """One figure per area: actual vs predicted for each model (stacked panels).

    Three areas x three model panels == the nine prediction plots required by
    Task 3.II.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        frame = res.to_frame()
        ax.plot(frame.index, frame["y_true"], color="black", linewidth=1.0,
                label="actual")
        ax.plot(frame.index, frame["y_pred"],
                color=_MODEL_COLOURS.get(res.model_name, "crimson"),
                linewidth=1.1, linestyle="--", label=f"{res.model_name} forecast")
        ax.set_ylabel("traffic")
        ax.set_title(f"{res.model_name} -- one-step-ahead forecast")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Date (test week 16-22 Dec)")
    fig.suptitle(f"Forecasts -- {area_label} (Square id {square_id})", y=1.005)
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


def plot_training_curves(results: list[ForecastResult]):
    """Plot train/validation loss curves for the neural models."""
    neural = [r for r in results if "history" in r.extra]
    if not neural:
        return None
    fig, axes = plt.subplots(1, len(neural), figsize=(6 * len(neural), 4),
                             squeeze=False)
    for ax, res in zip(axes[0], neural):
        hist = res.extra["history"]
        ax.plot(hist["loss"], label="train loss")
        ax.plot(hist["val_loss"], label="val loss")
        ax.set_title(f"{res.model_name} training (area {res.square_id})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.legend()
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig


def plot_error_over_time(area_label: str, results: list[ForecastResult]):
    """Plot absolute error across the test week -- used for failure analysis."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for res in results:
        frame = res.to_frame()
        err = np.abs(frame["y_true"] - frame["y_pred"])
        ax.plot(frame.index, err, linewidth=1.0,
                color=_MODEL_COLOURS.get(res.model_name, None),
                label=f"{res.model_name}")
    ax.set_title(f"Absolute forecast error over the test week -- {area_label}")
    ax.set_xlabel("Date (test week 16-22 Dec)")
    ax.set_ylabel("|actual - predicted|")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)        # prevent double-display in notebooks; savefig still works
    return fig
