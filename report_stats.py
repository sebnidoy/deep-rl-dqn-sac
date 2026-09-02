"""Derived statistics used in the HW3 write-up."""
from pathlib import Path

import numpy as np

from summarize_runs import load, series

EXP = Path("exp")

RUNS = {
    "hc_fixed": "HalfCheetah-v4_sac_sd1_20260826_000805",
    "hc_auto": "HalfCheetah-v4_sac_autotune_sd1_20260826_235627",
    "hop_single": "Hopper-v4_sac_singleq_sd1_20260824_162708",
    "hop_clip": "Hopper-v4_sac_clipq_sd1_20260824_162708",
    "mspacman": "MsPacman_dqn_sd1_20260825_222957",
    "ll_base": "LunarLander-v2_dqn_sd1_20260823_165741",
}

data = {k: load(EXP / v) for k, v in RUNS.items()}


def stats(name, key="Eval_AverageReturn", tail_frac=0.2):
    s = series(data[name], key)
    steps = np.array([x for x, _ in s])
    vals = np.array([v for _, v in s])
    cut = steps.max() * (1 - tail_frac)
    tail = vals[steps >= cut]
    return steps, vals, tail


print("--- HalfCheetah fixed vs autotune (eval return) ---")
for k in ("hc_fixed", "hc_auto"):
    st, v, tail = stats(k)
    print(
        f"{k:10s} max={v.max():9.1f}@{st[v.argmax()]:>7d}  "
        f"mean={v.mean():8.1f}  last20%mean={tail.mean():8.1f}  final={v[-1]:8.1f}"
    )

print("\n--- learned alpha trajectory (HalfCheetah autotune) ---")
s = series(data["hc_auto"], "alpha")
st = np.array([x for x, _ in s])
al = np.array([v for _, v in s])
for frac in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
    i = min(int(frac * (len(al) - 1)), len(al) - 1)
    print(f"  step {st[i]:>7d}  alpha={al[i]:.4f}")
print(f"  min alpha={al.min():.4f}@{st[al.argmin()]}  max={al.max():.4f}@{st[al.argmax()]}")

print("\n--- entropy (HalfCheetah) ---")
for k in ("hc_fixed", "hc_auto"):
    s = series(data[k], "entropy")
    v = np.array([x for _, x in s])
    print(f"{k:10s} first={v[0]:7.3f} min={v.min():7.3f} last={v[-1]:7.3f} last10mean={v[-10:].mean():7.3f}")
print("  target_entropy for HalfCheetah = -6 (action dim 6)")

print("\n--- Hopper single-Q vs clipped double-Q ---")
for k in ("hop_single", "hop_clip"):
    st, v, tail = stats(k)
    q = np.array([x for _, x in series(data[k], "q_values")])
    print(
        f"{k:11s} evalmax={v.max():8.1f}@{st[v.argmax()]:>7d} evalmean={v.mean():7.1f} "
        f"last20%={tail.mean():7.1f} | qmax={q.max():7.1f} qlast={q[-1]:7.1f}"
    )
    n_1500 = (v >= 1500).sum()
    print(f"             evals>=1500: {n_1500}/{len(v)}")

print("\n--- Hopper: logged Q vs rough on-policy discounted value ---")
GAMMA, BETA = 0.99, 0.1
for k in ("hop_single", "hop_clip"):
    rows = data[k]
    ret = dict(series(rows, "Eval_AverageReturn"))
    eplen = dict(series(rows, "Eval_AverageEpLen"))
    q = dict(series(rows, "q_values"))
    ent = dict(series(rows, "entropy"))
    print(f"  {k}")
    for step in (100000, 200000, 300000, 400000):
        L, R = eplen[step], ret[step]
        horizon = (1 - GAMMA**L) / (1 - GAMMA)
        nq = min(q, key=lambda s: abs(s - step))
        ne = min(ent, key=lambda s: abs(s - step))
        # discounted value of an L-step rollout at the average per-step reward,
        # including the entropy bonus that is backed up into the critic target
        v_est = (R / L) * horizon + BETA * ent[ne] * horizon
        print(
            f"    step {step:>7d}: return={R:7.1f} eplen={L:6.1f} "
            f"V_est={v_est:7.1f} Q_logged={q[nq]:7.1f} ratio={q[nq] / v_est:5.2f}"
        )

print("\n--- MsPacman train vs eval by window ---")
tr = series(data["mspacman"], "Train_EpisodeReturn")
ev = series(data["mspacman"], "Eval_AverageReturn")
eps = series(data["mspacman"], "epsilon")
print("  first evals:", [(s, round(v)) for s, v in ev[:6]])
windows = [
    (0, 20000), (20000, 50000), (50000, 100000), (100000, 200000),
    (200000, 400000), (400000, 700000), (700000, 1000000),
]
for lo, hi in windows:
    m = lambda xs: np.mean(xs) if xs else float("nan")
    print(
        f"  {lo:>7d}-{hi:<7d} train={m([v for s_, v in tr if lo <= s_ < hi]):7.1f} "
        f"eval={m([v for s_, v in ev if lo <= s_ < hi]):7.1f} "
        f"eps={m([v for s_, v in eps if lo <= s_ < hi]):.3f}"
    )

print("\n--- LunarLander LR sweep ---")
sweep = {
    "1e-4": "LunarLander-v2_dqn_lr1e-4_sd1_20260823_172521",
    "3e-4": "LunarLander-v2_dqn_lr3e-4_sd1_20260823_175626",
    "1e-3 (baseline)": "LunarLander-v2_dqn_sd1_20260823_165741",
    "3e-3": "LunarLander-v2_dqn_lr3e-3_sd1_20260823_182311",
}
for label, run in sweep.items():
    rows = load(EXP / run)
    s = series(rows, "Eval_AverageReturn")
    st = np.array([x for x, _ in s])
    v = np.array([x for _, x in s])
    q = np.array([x for _, x in series(rows, "q_values")])
    hits = st[v >= 200]
    print(
        f"  {label:16s} max={v.max():7.1f}@{st[v.argmax()]:>7d} n>=200={len(hits):2d} "
        f"first>=200={hits[0] if len(hits) else '-':>7} last20%={v[st >= 0.8 * st.max()].mean():7.1f} qmax={q.max():6.1f}"
    )
