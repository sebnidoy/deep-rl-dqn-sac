"""Regenerate the LunarLander learning-rate sweep figure with readable legend labels."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_runs import load, series

EXP = Path("exp")

RUNS = [
    ("lr = 1e-4", "LunarLander-v2_dqn_lr1e-4_sd1_20260823_172521", "tab:blue"),
    ("lr = 3e-4", "LunarLander-v2_dqn_lr3e-4_sd1_20260823_175626", "tab:orange"),
    ("lr = 1e-3 (baseline)", "LunarLander-v2_dqn_sd1_20260823_165741", "tab:green"),
    ("lr = 3e-3", "LunarLander-v2_dqn_lr3e-3_sd1_20260823_182311", "tab:red"),
]

fig, ax = plt.subplots(figsize=(10, 6))
for label, run, color in RUNS:
    s = series(load(EXP / run), "Eval_AverageReturn")
    ax.plot(
        [x for x, _ in s], [v for _, v in s],
        marker="o", markersize=3, linewidth=1.4, color=color, label=label,
    )

ax.axhline(200, color="gray", linestyle="--", linewidth=1, label="Target (200)")
ax.set_xlabel("Environment steps")
ax.set_ylabel("Evaluation return")
ax.set_title("LunarLander-v2 Double DQN: learning-rate sweep")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

fig.savefig(EXP / "lunarlander_lr_sweep.png", dpi=150, bbox_inches="tight")
print("wrote", EXP / "lunarlander_lr_sweep.png")
