"""Figure for E12 (measured load and PV profiles). Run after rev_e12_measured.py."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vfem_core import C, P
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
T = json.load(open("results/e12_traj.json"))
d = np.array(T["d30"]); n = len(d)
idx = pd.date_range(T["index0"], periods=n, freq="30min")

fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.12, wspace=0.28)
ax = fig.add_subplot(gs[:, 0])
ax.plot(np.arange(n) * 0.5, np.sort(d)[::-1], color=C["neutral"], lw=1.6)
ax.axhline(P.d_hi, color="k", ls=":", lw=1.0); ax.axhline(P.d_lo, color="k", ls=":", lw=1.0)
ax.axhline(0, color="k", lw=0.6)
ax.text(n * 0.5 * 0.98, P.d_hi - 0.12, r"$d_{hi}$", ha="right", va="top", fontsize=8)
ax.text(n * 0.5 * 0.02, P.d_lo + 0.12, r"$d_{lo}$", ha="left", va="bottom", fontsize=8)
ax.set_xlabel("hours of the year, sorted"); ax.set_ylabel("measured net deficit $d$ [kW]")
ax.set_title("Duration curve", fontsize=9)

k = int(np.argmax(d)); a = max(0, k - 3 * 48); b = min(n, k + 3 * 48)
tt = (np.arange(a, b) - a) * 0.5 / 24
ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax1.step(tt, d[a:b], where="post", color=C["neutral"], lw=1.0)
ax1.axhline(P.d_hi, color="k", ls=":", lw=1.0)
ax1.set_ylabel("$d$ [kW]"); ax1.set_title(f"Six days around the annual peak ({idx[k]:%d %b %Y})", fontsize=9)
plt.setp(ax1.get_xticklabels(), visible=False)
lab = {"bb0.5": ("viable bang-bang", C["viable"]), "fr10": ("fixed reserve, 10 kWh", C["reserve"]),
       "mpc|2|nominal|F|P": ("economic MPC, persistence", C["mpc"])}
for key, (name, col) in lab.items():
    ax2.plot(tt, np.array(T["traj"][key]["s"])[a:b], color=col, lw=1.2, label=name)
ax2.set_ylabel("SoC $s$ [kWh]"); ax2.set_xlabel("time [days]"); ax2.set_ylim(-0.5, P.S_max + 0.5)
h, l = ax2.get_legend_handles_labels()
leg = fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09), frameon=False)
for ln in leg.get_lines():
    ln.set_linewidth(2.2)
fig.savefig(f"{FIG}/fig_measured.pdf", bbox_inches="tight"); plt.close(fig)
print("ok", idx[k])
