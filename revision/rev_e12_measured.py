"""Experiment E12: measured load and PV profiles.

One year (1 July 2011 - 30 June 2012) of half-hourly measured household consumption (GC) and
gross PV generation (GG) of customer 12 of the Ausgrid "Solar home electricity data"
(300 homes, Sydney region; CC BY). The two channels are scaled by constant factors so that the
net deficit d = a*GC - b*GG spans exactly the admissible range [d_lo, d_hi] of the paper over
the year, and are held constant over each half hour (zero-order hold) at the EMS period.

Part (a): all policies over the full year on the nominal plant.
Part (b): the bound d_hi is estimated from the first three months only and the policies are
          tested on the remaining nine months, with and without kernel inflation.
"""
import json, os, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd
from vfem_ext import *

OUT = "results"; os.makedirs(OUT, exist_ok=True)
DATA = "data/ausgrid_customer12_2011-2012.csv"
HALF = 0.5                                             # h, resolution of the measurements


def load_measured(p=P):
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    L, G = df["GC"].to_numpy(float), df["GG"].to_numpy(float)
    a = b = 1.0
    for _ in range(500):                               # fixed point: max d = d_hi, min d = d_lo
        d = a * L - b * G
        a *= p.d_hi / d.max()
        d = a * L - b * G
        i = int(np.argmin(d))
        b = (a * L[i] - p.d_lo) / G[i]
    d = a * L - b * G
    assert abs(d.max() - p.d_hi) < 1e-9 and abs(d.min() - p.d_lo) < 1e-9
    return df.index, a * L, b * G, d, a, b


IDX, LOAD, PV, D30, A, B = load_measured()
PER = int(round(HALF / P.dt))
DS = np.repeat(D30, PER)                               # zero-order hold at the EMS period


def simulate_rec(policy, ds, p, s0=CFG.s0, g0=CFG.g0, ckpt=None, seg_days=61):
    """Same plant step as simulate_x (step_x); also returns per-step shedding.

    With ckpt, the run is cut into segments of seg_days and the state is saved after each one,
    so that an interrupted run resumes. Segment boundaries coincide with MPC re-planning
    instants, so the result does not depend on the segmentation."""
    n = len(ds)
    seg = int(round(seg_days * 24 / p.dt))
    S = np.empty(n); Gd = np.empty(n); SH = np.empty(n)
    st = dict(k=0, s=s0, g=g0, fuel=0.0, thr=0.0, on_time=0.0, starts=0, prev_on=bool(g0 > TOL))
    if ckpt and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        st = z["st"].item(); k0 = st["k"]
        S[:k0] = z["S"]; Gd[:k0] = z["G"]; SH[:k0] = z["SH"]
    while st["k"] < n:
        if hasattr(policy, "reset"):
            policy.reset()
        k0 = st["k"]; k1 = min(n, k0 + seg)
        s, g = st["s"], st["g"]
        for k in range(k0, k1):
            u = policy(s, g, ds[k], k, ds, p)
            s, g, sh, th = step_x(s, g, u, ds[k], p)
            S[k] = s; Gd[k] = g; SH[k] = sh
            st["fuel"] += g * p.dt; st["thr"] += th
            on = g > TOL
            st["on_time"] += p.dt if on else 0.0
            st["starts"] += int(on and not st["prev_on"]); st["prev_on"] = bool(on)
        st.update(k=k1, s=s, g=g)
        if ckpt:
            np.savez(ckpt, st=np.array(st, dtype=object), S=S[:k1], G=Gd[:k1], SH=SH[:k1])
            print(f"  [{os.path.basename(ckpt)}] {100*k1/n:5.1f}%", flush=True)
    return dict(fuel=st["fuel"], shed=float(SH.sum()), thr=st["thr"], on_time=st["on_time"],
                starts=st["starts"], s=S, g=Gd, sh=SH)


def assumed(pol, p_assumed):
    """Policy that evaluates its kernel with assumed parameters while the plant uses the true ones."""
    def f(s, g, d, k, ds, p):
        return pol(s, g, d, k, ds, p_assumed)
    return f


POLICIES = [
    ("Efficiency-greedy",                    "greedy"),
    ("Fixed reserve, 8 kWh",                 "fr8"),
    ("Fixed reserve, 10 kWh",                "fr10"),
    ("Viable bang-bang",                     "bb0.5"),
    ("Viable proportional",                  "prop1"),
    ("Viable, state-dependent trigger",      "bbs0"),
    ("Economic MPC, persistence, H=2 h",     "mpc|2|nominal|F|P"),
    ("  + terminal viability",               "mpc|2|nominal|T|P"),
    ("Robust MPC, persistence, H=2 h",       "mpc|2|robust|F|P"),
    ("Chance-constrained MPC, persistence, H=2 h", "mpc|2|chance|F|P"),
    ("Worst-case-tube MPC, H=2 h",           "mpc|2|worst|F|P"),
    ("Economic MPC, perfect forecast, H=4 h", "mpc|4|nominal|F|N"),
    ("  + terminal viability ",              "mpc|4|nominal|T|N"),
]


def make(name):
    if name == "greedy": return greedy
    if name.startswith("fr"): return fixed_reserve(float(name[2:]))
    if name.startswith("bbs"): return bb_state(float(name[3:]))
    if name.startswith("bb"): return viable_bangbang(float(name[2:]))
    if name.startswith("prop"): return viable_proportional(float(name[4:]))
    _, H, mode, term, pers = name.split("|")
    kw = dict(mode=mode, terminal=term == "T", persistence=pers == "P")
    if mode == "robust": kw["w"] = 1.5
    if mode == "chance": kw.update(sigma_f=0.75, eps=0.05)
    return mpc_x(float(H), **kw)


def summarise(label, key, r, ds, p):
    per_day = int(round(24 / p.dt))
    nd = len(ds) // per_day
    day_shed = r["sh"][:nd * per_day].reshape(nd, per_day).sum(1)
    mu = margin(r["s"], r["g"], p)
    return dict(policy=label.strip(), key=key, energy=r["fuel"], shed=r["shed"],
                days_with_shed=int((day_shed > 1e-6).sum()), days=nd,
                steps_with_shed=int((r["sh"] > 1e-9).sum()),
                max_daily_shed=float(day_shed.max()), min_mu=float(mu.min()),
                starts=int(r["starts"]), on_time=r["on_time"], throughput=r["thr"],
                efc=r["thr"] / p.S_max, affine=affine_cost(r))


def _job(args):
    label, key = args
    os.makedirs(f"{OUT}/e12_ckpt", exist_ok=True)
    ck = f"{OUT}/e12_ckpt/" + key.replace("|", "_") + ".npz" if key.startswith("mpc") else None
    r = simulate_rec(make(key), DS, PXN, ckpt=ck)
    out = summarise(label, key, r, DS, PXN)
    print(f"done {key:22s} energy {out['energy']:8.1f}  shed {out['shed']:7.2f}  "
          f"days {out['days_with_shed']}", flush=True)
    keep = {k: r[k][::PER] for k in ("s", "g")} if key in ("bb0.5", "fr10", "mpc|2|nominal|F|P") else None
    return out, key, keep


def describe():
    d = D30
    jump = np.abs(np.diff(d))
    run = best = 0
    for x in d > 3.0:
        run = run + 1 if x else 0
        best = max(best, run)
    return dict(source="Ausgrid Solar home electricity data, customer 12, 2011-07-01 to 2012-06-30",
                n_half_hours=int(len(d)), scale_load=A, scale_pv=B,
                load_mean_kW=float(LOAD.mean()), load_max_kW=float(LOAD.max()),
                pv_max_kW=float(PV.max()), load_kWh=float(LOAD.sum() * HALF),
                pv_kWh=float(PV.sum() * HALF),
                d_mean=float(d.mean()), d_max=float(d.max()), d_min=float(d.min()),
                frac_deficit=float((d > 0).mean()), d_p99=float(np.percentile(d, 99)),
                half_hours_above_4kW=int((d > 4.0).sum()), half_hours_above_3kW=int((d > 3.0).sum()),
                longest_run_above_3kW_h=best * HALF,
                largest_jump_kW=float(jump.max()), jumps_above_2kW=int((jump > 2.0).sum()),
                positive_deficit_kWh=float(np.clip(d, 0, None).sum() * HALF),
                surplus_kWh=float(np.clip(-d, 0, None).sum() * HALF))


def part_b():
    """Bound estimated from the first three months; test on the remaining nine."""
    cut = int((IDX < pd.Timestamp("2011-10-01")).sum())
    d_hist = float(D30[:cut].max())
    ds = np.repeat(D30[cut:], PER)
    exceed = int((D30[cut:] > d_hist + 1e-9).sum())
    p_ass = replace(PXN, d_hi=d_hist)
    p_inf = replace(PXN, d_hi=d_hist + 1.0)             # inflation eps_d = 1 kW, as in E5
    rows = []
    for label, pol, pk in [
        ("Viable bang-bang, estimated bound", assumed(viable_bangbang(0.5), p_ass), p_ass),
        ("Viable proportional, estimated bound", assumed(viable_proportional(1.0), p_ass), p_ass),
        ("Viable bang-bang, inflated kernel", assumed(viable_bangbang(0.5), p_inf), p_inf),
        ("Viable proportional, inflated kernel", assumed(viable_proportional(1.0), p_inf), p_inf),
        ("Viable bang-bang, true bound", viable_bangbang(0.5), PXN),
        ("Fixed reserve, 10 kWh", fixed_reserve(10.0), PXN),
    ]:
        r = simulate_rec(pol, ds, PXN)
        o = summarise(label, label, r, ds, PXN)
        o["assumed_d_hi"] = pk.d_hi
        rows.append(o)
        print(f"(b) {label:40s} energy {o['energy']:8.1f} shed {o['shed']:6.2f}", flush=True)
    return dict(history_end="2011-09-30", d_hi_estimated=d_hist, half_hours_exceeding=exceed,
                max_true=float(D30[cut:].max()), test_days=len(ds) * P.dt / 24), rows


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 1:                              # pre-compute checkpoints for given keys
        with Pool(2) as pool:
            pool.map(_job, [(k, k) for k in sys.argv[1:]], chunksize=1)
        sys.exit(0)
    desc = describe()
    print(json.dumps(desc, indent=1), flush=True)
    with Pool(2) as pool:
        res = pool.map(_job, POLICIES, chunksize=1)
    order = {k: i for i, (_, k) in enumerate(POLICIES)}
    res.sort(key=lambda x: order[x[1]])
    rows = [r[0] for r in res]
    traj = {k: {n: v.tolist() for n, v in keep.items()} for _, k, keep in res if keep}
    info_b, rows_b = part_b()
    pd.DataFrame(rows).to_csv(f"{OUT}/e12_measured.csv", index=False)
    pd.DataFrame(rows_b).to_csv(f"{OUT}/e12_measured_bound.csv", index=False)
    json.dump(dict(description=desc, full_year=rows, estimated_bound=info_b,
                   estimated_bound_rows=rows_b), open(f"{OUT}/e12_measured.json", "w"), indent=1)
    json.dump(dict(d30=D30.tolist(), traj=traj, index0=str(IDX[0])),
              open(f"{OUT}/e12_traj.json", "w"))
    print(f"total {time.time()-t0:.0f}s", flush=True)
