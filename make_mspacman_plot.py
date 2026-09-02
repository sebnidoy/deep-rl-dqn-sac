"""Regenerate the MsPacman train-vs-eval figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_runs import load, series

RUN = Path("exp/MsPacman_dqn_sd1_20260825_222957")
rows = load(RUN)

tr = series(rows, "Train_EpisodeReturn")
ev = series(rows, "Eval_AverageReturn")

tr_steps = np.array([s for s, _ in tr])
tr_vals = np.array([v for _, v in tr])
window = 50
smoothed = np.convolve(tr_vals, np.ones(window) / window, mode="valid")
smoothed_steps = tr_steps[window - 1:]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(smoothed_steps, smoothed, label=f"Train return ({window}-ep average)", linewidth=1.5)
ax.plot(
    [s for s, _ in ev], [v for _, v in ev],
    marker="o", markersize=3, linewidth=1.2, label="Eval return (greedy)",
)
ax.axhline(1500, color="gray", linestyle="--", linewidth=1, label="Target (~1500)")
ax.set_xlabel("Environment steps")
ax.set_ylabel("Return")
ax.set_title("MsPacman Double DQN: train (ε-greedy) vs. eval (greedy) return")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

fig.savefig(RUN / "train_vs_eval.png", dpi=150, bbox_inches="tight")
print("wrote", RUN / "train_vs_eval.png")
