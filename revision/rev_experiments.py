"""Revision experiments E7-E11 (abrupt drops, power limit, robust/chance-constrained MPC,
implementation uncertainty, sampled-data tightness, non-dominance). Seed 20250901."""
import json, os, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from vfem_ext import *

OUT = "results_rev"; os.makedirs(OUT, exist_ok=True)
SEED = CFG.seed
N_MC = 300
PB = replace(PXN, P_b=4.0)                         # battery/inverter limited plant


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k == n else float(beta_dist.ppf(1 - alpha, k + 1, n - k))


def ensembles():
    rng = np.random.default_rng(SEED)
    ou = [ou_series(rng, P) for _ in range(N_MC)]          # identical to E3
    rng2 = np.random.default_rng(SEED + 1)
    col = [collapse_series(rng2, P) for _ in range(N_MC)]
    rng3 = np.random.default_rng(SEED + 2)
    fl = [flicker_series(rng3, P) for _ in range(N_MC)]
    return {"ou": ou, "collapse": col, "flicker": fl}


ENS = ensembles()


def row(name, r, **extra):
    return dict(policy=name, energy=r["fuel"], shed=r["shed"], starts=r["starts"],
                on_time=r["on_time"], affine=affine_cost(r), throughput=r["thr"],
                efc=r["efc"], min_mu=r["min_mu"], **extra)


# ------------------------------------------------------------------ policy factories (picklable by name)
def make(name):
    if name == "greedy": return greedy
    if name.startswith("fr"): return fixed_reserve(float(name[2:]))
    if name.startswith("bbs"): return bb_state(float(name[3:]))
    if name.startswith("bbx"): return bb_x(float(name[3:]))
    if name.startswith("bb"): return viable_bangbang(float(name[2:]))
    if name.startswith("propx"): return prop_x(float(name[5:]))
    if name.startswith("prop"): return viable_proportional(float(name[4:]))
    if name.startswith("mpc"):
        # mpc|H|mode|terminal|persistence
        _, H, mode, term, pers = name.split("|")
        kw = dict(mode=mode, terminal=term == "T", persistence=pers == "P")
        if mode == "robust": kw["w"] = 1.5
        if mode == "chance": kw.update(sigma_f=0.75, eps=0.05)
        return mpc_x(float(H), **kw)
    raise ValueError(name)


def _mc_job(args):
    name, ens, plant = args
    p = {"nom": PXN, "pb": PB}[plant]
    pol = make(name)
    res = []
    for d in ENS[ens]:
        r = run(pol, d, p)
        res.append((r["fuel"], r["shed"], r["min_mu"], r["thr"], affine_cost(r)))
    return name, ens, plant, res


def mc_table(names, ens, plant="nom", procs=2):
    with Pool(procs) as pool:
        out = pool.map(_mc_job, [(n, ens, plant) for n in names])
    rows = []
    for name, e, pl, res in out:
        a = np.array(res)
        k = int((a[:, 1] > TOL).sum())
        rows.append(dict(policy=name, ensemble=e, plant=pl, p_shed=k / len(a),
                         cp_upper=cp_upper(k, len(a)), mean_shed=a[:, 1].mean(),
                         p95_shed=np.percentile(a[:, 1], 95), mean_energy=a[:, 0].mean(),
                         p95_energy=np.percentile(a[:, 0], 95), min_mu=a[:, 2].min(),
                         mean_thr=a[:, 3].mean(), mean_affine=a[:, 4].mean()))
    return pd.DataFrame(rows)


R = {}
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ================================================================== E7a power-limited kernel
def frac_x(p, n=200001):
    g = np.linspace(0, p.G_max, n)
    sm = np.clip(s_min(g, p), 0, p.S_max)
    sm = np.where(g < g_floor(p), p.S_max, sm)
    return 1 - np.trapezoid(sm, g) / (p.S_max * p.G_max)

pl_rows = []
for pb in (3.0, 4.0, 5.0, INF):
    p_ = replace(PXN, P_b=pb)
    pl_rows.append(dict(P_b=pb, g_floor=g_floor(p_), frac=frac_x(p_),
                        s_min_floor=float(s_min(g_floor(p_), p_))))
R["powerlimit_kernel"] = pl_rows
log("E7a kernel", pl_rows)

# ================================================================== E7 deterministic abrupt disturbances
det = {}
_, d_wc = worst_case_series(P)
_, d_step = step_drought(P)
_, d_rec = recurring_droughts(P)
det_series = {"ramped": d_wc, "step": d_step, "recurring": d_rec}
pols_det = ["greedy", "fr10", "bb0.5", "prop1.0"]
rows = []
for sname, d in det_series.items():
    for plant, p in (("nom", PXN), ("pb", PB)):
        names = pols_det + (["bbx0.5", "propx1.0"] if plant == "pb" else [])
        for n in names:
            s0 = None; g0 = None
            if plant == "pb" and n.endswith("x0.5") or n.endswith("x1.0"):
                g0 = max(CFG.g0, g_floor(p))
            r = run(make(n), d, p, g0=g0)
            rows.append(row(n, r, series=sname, plant=plant))
R["abrupt_det"] = rows
log("E7 det done")

# ================================================================== E8 MPC family, deterministic
mpc_rows = []
cases = [("perfect", 0.0, False), ("bias20", 0.20, False), ("bias30", 0.30, False),
         ("persistence", 0.0, True)]
for sname in ("ramped", "step"):
    d = det_series[sname]
    for cname, bias, pers in cases:
        for H in (1.0, 2.0, 4.0):
            for mode, term in (("nominal", False), ("nominal", True), ("robust", False),
                               ("chance", False), ("worst", False)):
                kw = dict(mode=mode, terminal=term, persistence=pers, bias=bias)
                if mode == "robust": kw["w"] = 1.5
                if mode == "chance": kw.update(sigma_f=0.75, eps=0.05)
                r = run(mpc_x(H, **kw), d, PXN)
                mpc_rows.append(row(f"{mode}{'+TV' if term else ''}", r, series=sname,
                                    forecast=cname, H=H))
    for n in ("bb0.5", "prop1.0", "fr10", "greedy"):
        r = run(make(n), d, PXN)
        mpc_rows.append(row(n, r, series=sname, forecast="none", H=0.0))
R["mpc_det"] = mpc_rows
log("E8 det done")

# ================================================================== E10 sampled-data tightness
sd_rows = []
for dt_ in (0.02, 0.05, 0.1, 0.2, 0.4):
    p_ = replace(PXN, dt=dt_)
    B = 2 * p_.d_hi * dt_
    Lx = B - p_.R * dt_**2
    # adversarial one-step test: g = R dt, mu just above mu_bar, d = d_hi
    def adv(mu_bar):
        g0 = p_.R * dt_
        s0 = float(s_min(g0, p_)) + mu_bar + 1e-9
        r = simulate_x(viable_bangbang(mu_bar), np.full(int(1.0 / dt_) + 2, p_.d_hi), p_,
                       s0=s0, g0=g0)
        return r["min_mu"]
    adv_below = adv(0.95 * Lx); adv_at = adv(Lx)
    _, dwc = worst_case_series(p_)
    _, dst = step_drought(p_)
    def min_mubar(d):
        lo, hi = 0.0, B * 1.5
        if simulate_x(viable_bangbang(0.0), d, p_)["shed"] <= TOL:
            return 0.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if simulate_x(viable_bangbang(m), d, p_)["shed"] <= TOL: hi = m
            else: lo = m
        return hi
    emp_wc, emp_st = min_mubar(dwc), min_mubar(dst)
    e_B = simulate_x(viable_bangbang(B), dwc, p_)["fuel"]
    e_emp = simulate_x(viable_bangbang(emp_wc), dwc, p_)["fuel"]
    e_state = simulate_x(bb_state(0.0), dwc, p_)
    sd_rows.append(dict(dt=dt_, bound=B, exact=Lx, slack_pct=100 * (B - Lx) / B,
                        adv_min_mu_below=adv_below, adv_min_mu_at=adv_at,
                        emp_min_mubar_ramped=emp_wc, emp_min_mubar_step=emp_st,
                        energy_at_bound=e_B, energy_at_emp=e_emp,
                        energy_state_trigger=e_state["fuel"], shed_state_trigger=e_state["shed"]))
R["sampled"] = sd_rows
log("E10 sampled done", pd.DataFrame(sd_rows).round(4).to_string())

# ================================================================== E9 implementation uncertainty (deterministic part)
imp_rows = []
for sname in ("ramped", "step"):
    d = det_series[sname]
    for lab, kw in [("ideal", {}), ("delay 1 step", dict(delay=1)), ("delay 3 steps", dict(delay=3)),
                    ("SoC over-read 0.5 kWh", dict(soc_bias=0.5)),
                    ("SoC noise 0.2 kWh", dict(soc_noise=0.2))]:
        for mub in (0.5, 1.5):
            rng = np.random.default_rng(SEED)
            r = simulate_x(viable_bangbang(mub), d, PXN, rng=rng, **kw)
            imp_rows.append(row(f"bb{mub}", r, series=sname, case=lab))
R["impl_det"] = imp_rows
log("E9 det done")

json.dump(R, open(f"{OUT}/det.json", "w"), indent=1, default=float)

# ================================================================== Monte-Carlo blocks
mc_names_ff = ["greedy", "fr6", "fr8", "fr10", "fr12", "bb0.0", "bb0.2", "bb0.5", "bb1.0",
               "bb2.0", "prop0.5", "prop1.0", "prop2.0", "bbs0.0"]
mc_names_mpc = ["mpc|2|nominal|F|P", "mpc|2|nominal|T|P", "mpc|2|robust|F|P",
                "mpc|2|chance|F|P", "mpc|2|worst|F|P"]
frames = []
for ens in ("ou", "collapse", "flicker"):
    frames.append(mc_table(mc_names_ff, ens)); log("MC ff", ens)
frames.append(mc_table(["greedy", "fr10", "bb0.5", "prop1.0", "bbx0.5", "propx1.0"],
                       "collapse", plant="pb")); log("MC pb")
pd.concat(frames).to_csv(f"{OUT}/mc_ff.csv", index=False)
frames_mpc = []
for ens in ("collapse", "ou"):
    frames_mpc.append(mc_table(mc_names_mpc, ens)); log("MC mpc", ens)
pd.concat(frames_mpc).to_csv(f"{OUT}/mc_mpc.csv", index=False)

# implementation uncertainty MC (ou ensemble)
imp_mc = []
for lab, kw in [("ideal", {}), ("delay 1 step", dict(delay=1)), ("delay 3 steps", dict(delay=3)),
                ("SoC over-read 0.5 kWh", dict(soc_bias=0.5)), ("SoC noise 0.2 kWh", dict(soc_noise=0.2))]:
    for mub in (0.5, 1.5):
        rng = np.random.default_rng(SEED)
        rs = [simulate_x(viable_bangbang(mub), d, PXN, rng=rng, **kw) for d in ENS["ou"]]
        k = sum(r["shed"] > TOL for r in rs)
        imp_mc.append(dict(case=lab, mu_bar=mub, p_shed=k / N_MC, cp_upper=cp_upper(k, N_MC),
                           mean_shed=np.mean([r["shed"] for r in rs]),
                           mean_energy=np.mean([r["fuel"] for r in rs]),
                           min_mu=np.min([r["min_mu"] for r in rs])))
pd.DataFrame(imp_mc).to_csv(f"{OUT}/impl_mc.csv", index=False)
log("all done")
