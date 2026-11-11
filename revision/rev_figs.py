import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vfem_ext import *
from vfem_core import C
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
FIG = "figs_rev"; os.makedirs(FIG, exist_ok=True)
R = json.load(open("results_rev/det.json"))
PB = replace(PXN, P_b=4.0)

# ---------------- Fig: abrupt drought on the power-limited plant
t, d = step_drought(PB)
ra = simulate_x(viable_bangbang(0.5), d, PB, record=True)
rb = simulate_x(bb_x(0.5), d, PB, record=True)
fig, axs = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True)
axs[0].plot(t, d, color=C["neutral"], lw=1.6); axs[0].set_ylabel("deficit $d$ [kW]")
axs[0].set_title(r"Abrupt drought, battery/inverter limit $P_b=4$ kW")
axs[1].plot(t, ra["s"], color=C["greedy"], lw=1.6, label=f"nominal-kernel bang-bang (sheds {ra['shed']:.2f} kWh)")
axs[1].plot(t, rb["s"], color=C["viable"], lw=1.6, label=f"power-limited kernel (sheds {rb['shed']:.2f} kWh)")
axs[1].set_ylabel("SoC $s$ [kWh]"); axs[1].legend(fontsize=7.5, loc="lower left")
axs[2].plot(t, ra["g"], color=C["greedy"], lw=1.6)
axs[2].plot(t, rb["g"], color=C["viable"], lw=1.6)
axs[2].axhline(g_floor(PB), color="k", ls=":", lw=1.0)
axs[2].text(11.8, g_floor(PB) + 0.15, r"floor $d_{hi}-P_b$", ha="right", fontsize=7.5)
axs[2].set_ylabel("dispatch $g$ [kW]"); axs[2].set_xlabel("time [h]")
fig.savefig(f"{FIG}/fig_abrupt.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: sampled-data tightness
sd = pd.DataFrame(R["sampled"])
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5))
axs[0].plot(sd.dt, sd.bound, "o-", color="k", label=r"bound $2d_{hi}\Delta t$ (Eq. 16)")
axs[0].plot(sd.dt, sd.exact, "s--", color=C["greedy"], label=r"exact worst case $2d_{hi}\Delta t-R\Delta t^2$")
axs[0].plot(sd.dt, sd.emp_min_mubar_ramped, "^-", color=C["viable"], label=r"smallest safe $\bar\mu$, ramped drought")
axs[0].plot(sd.dt, sd.emp_min_mubar_step, "v-", color=C["prop"], label=r"smallest safe $\bar\mu$, abrupt drought")
axs[0].set_xlabel(r"EMS period $\Delta t$ [h]"); axs[0].set_ylabel(r"protection margin $\bar\mu$ [kWh]")
axs[0].legend(fontsize=7.5); axs[0].set_title("Worst-case requirement and observed requirement")
w = 0.012
x = sd.dt.values
axs[1].bar(np.arange(len(x)) - 0.27, sd.energy_at_bound, 0.27, color="k", label=r"$\bar\mu=2d_{hi}\Delta t$")
axs[1].bar(np.arange(len(x)), sd.energy_state_trigger, 0.27, color=C["viable"], label="state-dependent trigger")
axs[1].bar(np.arange(len(x)) + 0.27, sd.energy_at_emp, 0.27, color=C["prop"], label=r"smallest safe $\bar\mu$ (hindsight)")
axs[1].set_xticks(np.arange(len(x))); axs[1].set_xticklabels([f"{v:g}" for v in x])
axs[1].set_xlabel(r"EMS period $\Delta t$ [h]"); axs[1].set_ylabel("dispatchable energy [kWh]")
axs[1].set_ylim(15, None); axs[1].legend(fontsize=7.5); axs[1].set_title("Energy cost of the margin (ramped drought)")
fig.savefig(f"{FIG}/fig_sampled.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: non-dominance (needs MC tables)
if os.path.exists("results_rev/mc_mpc.csv"):
    ff = pd.read_csv("results_rev/mc_ff.csv"); mp = pd.read_csv("results_rev/mc_mpc.csv")
    allp = pd.concat([ff[ff.plant == "nom"], mp])
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7))
    lab = {"ou": "OU drift with drought plateaus", "collapse": "abrupt renewable collapses",
           "flicker": "rapid surplus/deficit switching"}
    for ax, ens in zip(axs, ("ou", "collapse", "flicker")):
        sub = allp[allp.ensemble == ens].reset_index(drop=True)
        nd = set(nondominated(sub[["mean_energy", "mean_shed"]].values))
        for i, r in sub.iterrows():
            kind = ("greedy" if r.policy == "greedy" else "reserve" if r.policy.startswith("fr")
                    else "mpc" if r.policy.startswith("mpc") else "prop" if r.policy.startswith("prop") else "viable")
            ax.scatter(r.mean_energy, r.mean_shed + 1e-3, s=46 if i in nd else 22, color=C[kind],
                       edgecolor="k" if i in nd else "none", zorder=3)
        ax.set_yscale("log"); ax.set_xlabel("mean dispatchable energy [kWh]")
        ax.set_title(lab[ens], fontsize=9)
    axs[0].set_ylabel("mean unserved critical energy + $10^{-3}$ [kWh]")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=C[k], label=l) for k, l in
          (("greedy", "efficiency-greedy"), ("reserve", "fixed reserve"), ("viable", "viable bang-bang"),
           ("prop", "viable proportional"), ("mpc", "MPC family (persistence forecast)"))]
    hs.append(Line2D([], [], marker="o", ls="", color="w", markeredgecolor="k", label="non-dominated"))
    axs[2].legend(handles=hs, fontsize=7, loc="upper right")
    fig.savefig(f"{FIG}/fig_nondom.pdf", bbox_inches="tight"); plt.close(fig)
print("figs done")
