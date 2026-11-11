# ==== CELL 1
import json
import os
from dataclasses import dataclass, replace, asdict, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import beta as beta_dist

rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

FIGDIR, RESDIR = "Figures", "results"
import matplotlib; matplotlib.use("Agg")
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(RESDIR, exist_ok=True)

C = {"greedy": "#C0504D", "viable": "#2E5496", "prop": "#4E9A51",
     "mpc": "#B07AA1", "mpcv": "#7B4F9D", "reserve": "#E08214", "neutral": "#7F7F7F"}

TOL = 1e-9

# ==== CELL 3
@dataclass(frozen=True)
class Params:
    """Plant parameters. Defaults are the nominal design of the paper."""
    S_max: float = 20.0     # kWh   usable battery capacity
    G_max: float = 6.0      # kW    dispatchable maximum (sizing requires >= d_hi)
    R:     float = 1.5      # kW/h  ramp limit
    d_hi:  float = 5.0      # kW    worst-case net deficit
    d_lo:  float = -3.0     # kW    best case (renewable surplus)
    dt:    float = 0.02     # h     EMS sampling / integration step
    # --- refinements (Prop. 3); defaults recover the nominal model ---
    eta:   float = 1.0      # -     discharge efficiency
    tau:   float = 0.0      # h     generator start-up dead time
    s_dod: float = 0.0      # kWh   unusable floor (depth-of-discharge limit)
    L_nc:  float = 0.0      # kW    curtailable non-critical load included in d
    T_d:   float = np.inf   # h     maximum drought duration (finite-horizon variant)


@dataclass(frozen=True)
class Cfg:
    """Experiment configuration: everything not in the plant."""
    seed: int = 20250901
    # worst-case tube
    T_wc: float = 12.0
    s0: float = 10.0
    g0: float = 1.0
    drought_start: float = 2.5
    drought_end: float = 7.5
    # Monte-Carlo / OU disturbance  (C1: these are now stated in the paper)
    n_mc: int = 300
    T_mc: float = 24.0
    ou_theta: float = 1.2        # mean-reversion rate [1/h]
    ou_sigma: float = 1.1        # volatility [kW/sqrt(h)]
    ou_mean_offset: float = 1.0  # kW, seasonal mean level
    ou_mean_amp: float = 1.5     # kW, seasonal amplitude
    ou_period: float = 24.0      # h
    n_plateau: tuple = (1, 2)    # inclusive range of drought plateaus per scenario
    plateau_dur: tuple = (2.0, 6.0)   # h
    # policies
    mu_bar_bb: float = 0.5
    mu_bar_prop: float = 1.0
    kp_prop: float = 3.0
    # MPC
    mpc_block: float = 0.5       # h, decision block length
    mpc_nu: int = 5              # ramp levels per block
    mpc_ns: int = 61             # DP grid in s
    mpc_ng: int = 31             # DP grid in g
    mpc_shed_penalty: float = 1e3
    # affine genset cost
    c_marginal: float = 1.0
    c_noload: float = 0.6        # kW-equivalent while running
    c_start: float = 2.0         # kWh-equivalent per start


P, CFG = Params(), Cfg()
assert P.G_max >= P.d_hi, "V is empty unless G_max >= d_hi (Prop. 1, remark 2)"

with open(f"{RESDIR}/parameters.json", "w") as f:
    json.dump({"params": asdict(P), "cfg": {k: list(v) if isinstance(v, tuple) else v
                                            for k, v in asdict(CFG).items()}},
              f, indent=2, default=str)
print(json.dumps({"params": asdict(P)}, indent=2, default=str))

# ==== CELL 5
def s_min(g, p: Params):
    """Minimal viable reserve at dispatch g (Prop. 1 nominally, Prop. 3 in general).

    delta        deficit the actuator must still cover
    dead         energy drawn during the start-up dead time tau      (Prop. 3c)
    ramp         energy drawn while ramping, truncated at T_d        (Prop. 3d)
    /eta         discharge losses                                     (Prop. 3a)
    + s_dod      unusable floor                                       (Prop. 3b)
    L_nc         curtailable load removed from the deficit            (Prop. 3e)
    """
    g = np.asarray(g, dtype=float)
    delta = np.maximum(p.d_hi - p.L_nc - g, 0.0)
    t_end = np.minimum(p.T_d, delta / p.R)
    ramp = delta * t_end - 0.5 * p.R * t_end**2
    return p.s_dod + (p.tau * delta + ramp) / p.eta


def margin(s, g, p: Params):
    """Viability margin. V = {mu >= 0}; mu is a control barrier certificate for V."""
    return s - s_min(g, p)


def viable_fraction(p: Params, n=200001):
    """|V|/|K| by direct integration of the actual boundary (C6).

    Valid for every refinement, and handles truncation by S_max.
    """
    g = np.linspace(0, p.G_max, n)
    nonviable = np.trapezoid(np.clip(s_min(g, p), 0, p.S_max), g)
    return 1 - nonviable / (p.S_max * p.G_max)


def viable_fraction_closed_form(p: Params):
    """Nominal closed form (Eq. 11) and whether its validity condition holds.

    Only meaningful for the nominal model: eta=1, tau=0, s_dod=0, L_nc=0, T_d=inf.
    """
    nominal = (p.eta == 1 and p.tau == 0 and p.s_dod == 0
               and p.L_nc == 0 and np.isinf(p.T_d))
    valid = nominal and p.R >= p.d_hi**2 / (2 * p.S_max)
    return 1 - p.d_hi**3 / (6 * p.R * p.S_max * p.G_max), valid


frac_cf, cf_valid = viable_fraction_closed_form(P)
frac = viable_fraction(P)
print(f"viable fraction: integral {frac:.4f}   closed form {frac_cf:.4f} "
      f"(valid: {cf_valid})")
print(f"validity condition R >= d_hi^2/(2 S_max) = {P.d_hi**2/(2*P.S_max):.3f} kW/h"
      f"   [nominal R = {P.R}]")
print(f"s_min(0) = {s_min(0.0, P):.3f} kWh ({100*s_min(0.0,P)/P.S_max:.0f}% of battery)")
assert abs(frac - frac_cf) < 1e-4, "integral and closed form must agree nominally"

# ==== CELL 7
def admissible_u(u, g, p: Params):
    """Project a requested ramp onto U(g) (C5)."""
    lo = -p.R if g > TOL else 0.0
    hi = +p.R if g < p.G_max - TOL else 0.0
    return float(np.clip(u, lo, hi))


def step_state(s, g, u, d, p: Params):
    """One exact zero-order-hold step. Returns (s', g', u_eff, shed_kWh).

    u is projected onto U(g) first; if the ramp would still cross a generator bound
    within the step, u_eff is the average realised ramp, which keeps the quadratic
    term consistent with the trajectory actually followed.
    """
    u = admissible_u(u, g, p)
    g_new = min(max(g + u * p.dt, 0.0), p.G_max)
    u_eff = (g_new - g) / p.dt

    de = (g - d) * p.dt + 0.5 * u_eff * p.dt**2      # exact ZOH energy increment
    if de < 0.0:
        de /= p.eta                                   # discharge losses only (C7)
    s_new = s + de

    shed = 0.0
    if s_new < p.s_dod:
        shed = p.s_dod - s_new                        # unserved critical energy [kWh]
        s_new = p.s_dod
    s_new = min(s_new, p.S_max)                       # surplus is curtailed
    return s_new, g_new, u_eff, shed


def simulate(policy, d_series, p: Params, cfg: Cfg = CFG, s0=None, g0=None, record=False):
    """Roll the plant forward. policy(s, g, d, k, d_series, p) -> requested ramp."""
    s = cfg.s0 if s0 is None else s0
    g = cfg.g0 if g0 is None else g0
    fuel = shed = on_time = 0.0
    starts = 0
    prev_on = g > TOL
    hist = {k: [] for k in ("s", "g", "mu", "shed_kW")}

    for k, d in enumerate(d_series):
        u = policy(s, g, d, k, d_series, p)
        s, g, _, sh = step_state(s, g, u, d, p)
        shed += sh
        fuel += g * p.dt
        on = g > TOL
        on_time += p.dt if on else 0.0
        starts += int(on and not prev_on)
        prev_on = on
        if record:
            hist["s"].append(s); hist["g"].append(g)
            hist["mu"].append(margin(s, g, p)); hist["shed_kW"].append(sh / p.dt)

    out = dict(fuel=fuel, shed=shed, on_time=on_time, starts=starts)
    if record:
        out |= {k: np.array(v) for k, v in hist.items()}
        out["min_mu"] = out["mu"].min()
    return out


def affine_cost(r, cfg: Cfg = CFG):
    """Dispatchable operating cost beyond raw energy (M5): marginal + no-load + starts."""
    return (cfg.c_marginal * r["fuel"] + cfg.c_noload * r["on_time"]
            + cfg.c_start * r["starts"])

# ==== CELL 9
def worst_case_series(p: Params, cfg: Cfg = CFG):
    """Bounded tube: surplus, one-hour ramp-in, five-hour drought at d_hi, recovery."""
    t = np.arange(0, cfg.T_wc, p.dt)
    d = np.full_like(t, p.d_lo)
    a, b = cfg.drought_start, cfg.drought_end
    up = (t >= a - 1.0) & (t < a)
    d[up] = p.d_lo + (p.d_hi - p.d_lo) * (t[up] - (a - 1.0))
    d[(t >= a) & (t < b)] = p.d_hi
    dn = (t >= b) & (t < b + 1.0)
    d[dn] = p.d_hi - (p.d_hi - p.d_lo) * (t[dn] - b)
    return t, np.clip(d, p.d_lo, p.d_hi)


def ou_series(rng, p: Params, cfg: Cfg = CFG):
    """Clipped Ornstein-Uhlenbeck deficit about a slow seasonal mean, plus droughts.

        dd = theta (m(t) - d) dt + sigma dW,
        m(t) = ou_mean_offset + ou_mean_amp * sin(2 pi t / ou_period),

    then 1-2 plateaus of 2-6 h clamped to d_hi, finally clipped to [d_lo, d_hi].
    """
    n = int(cfg.T_mc / p.dt)
    t = np.arange(n) * p.dt
    m = cfg.ou_mean_offset + cfg.ou_mean_amp * np.sin(2 * np.pi * t / cfg.ou_period)
    d = np.empty(n); d[0] = m[0]
    for k in range(1, n):
        d[k] = (d[k-1] + cfg.ou_theta * (m[k] - d[k-1]) * p.dt
                + cfg.ou_sigma * np.sqrt(p.dt) * rng.normal())
    for _ in range(rng.integers(cfg.n_plateau[0], cfg.n_plateau[1] + 1)):
        dur = rng.uniform(*cfg.plateau_dur)
        t0 = rng.uniform(0, cfg.T_mc - dur)
        d[(t >= t0) & (t < t0 + dur)] = p.d_hi
    return np.clip(d, p.d_lo, p.d_hi)

# ==== CELL 11
def greedy(s, g, d, k, ds, p):
    """Efficiency-greedy (Pareto-only): always ramp down."""
    return -p.R


def viable_bangbang(mu_bar):
    """Eq. (13). Minimal tangency-compatible viable feedback."""
    def pol(s, g, d, k, ds, p):
        return +p.R if margin(s, g, p) <= mu_bar else -p.R
    return pol


def viable_proportional(mu_bar, kp=CFG.kp_prop):
    """Eq. (14). Regulates the margin to a positive setpoint; stays interior to V."""
    def pol(s, g, d, k, ds, p):
        return kp * (mu_bar - margin(s, g, p))
    return pol


def fixed_reserve(s_res):
    """State-of-practice: a CONSTANT energy reserve, blind to the actuator state."""
    def pol(s, g, d, k, ds, p):
        return +p.R if s <= s_res else -p.R
    return pol

# ==== CELL 13
def _block_transition(p: Params, cfg: Cfg, d_block, s_grid, g_grid, u_levels):
    """Simulate one decision block from every grid node under every ramp level.

    Returns (s_next, g_next, stage_cost) each of shape (ns, ng, nu), using step_state
    so the MPC predictor and the plant share one integrator (N3).
    """
    ns, ng, nu = len(s_grid), len(g_grid), len(u_levels)
    S = np.repeat(s_grid[:, None, None], ng, 1).repeat(nu, 2).astype(float)
    G = np.repeat(g_grid[None, :, None], ns, 0).repeat(nu, 2).astype(float)
    U = np.repeat(u_levels[None, None, :], ns, 0).repeat(ng, 1).astype(float)
    cost = np.zeros_like(S)

    for d in d_block:
        u = np.clip(U, np.where(G > TOL, -p.R, 0.0), np.where(G < p.G_max - TOL, p.R, 0.0))
        G_new = np.clip(G + u * p.dt, 0.0, p.G_max)
        u_eff = (G_new - G) / p.dt
        de = (G - d) * p.dt + 0.5 * u_eff * p.dt**2
        de = np.where(de < 0, de / p.eta, de)
        S_new = S + de
        shed = np.maximum(p.s_dod - S_new, 0.0)
        S_new = np.minimum(np.maximum(S_new, p.s_dod), p.S_max)
        cost += G_new * p.dt + cfg.mpc_shed_penalty * shed
        S, G = S_new, G_new
    return S, G, cost


def _interp_cost(J, s_grid, g_grid, S, G):
    """Bilinear interpolation of the cost-to-go J at arbitrary (S, G)."""
    ds = s_grid[1] - s_grid[0]; dg = g_grid[1] - g_grid[0]
    a = np.clip((S - s_grid[0]) / ds, 0, len(s_grid) - 1.001)
    b = np.clip((G - g_grid[0]) / dg, 0, len(g_grid) - 1.001)
    i0 = a.astype(int); j0 = b.astype(int)
    fa = a - i0; fb = b - j0
    return ((1-fa)*(1-fb)*J[i0, j0] + (1-fa)*fb*J[i0, j0+1]
            + fa*(1-fb)*J[i0+1, j0] + fa*fb*J[i0+1, j0+1])


def mpc_policy(H, forecast_bias=0.0, terminal_viability=False, cfg: Cfg = CFG):
    """Receding-horizon economic MPC; see the markdown above."""
    state = {"t_next": -1.0, "plan": None}

    def solve(s, g, k, ds, p):
        nb = max(1, int(round(H / cfg.mpc_block)))
        per = max(1, int(round(cfg.mpc_block / p.dt)))
        s_grid = np.linspace(p.s_dod, p.S_max, cfg.mpc_ns)
        g_grid = np.linspace(0.0, p.G_max, cfg.mpc_ng)
        u_lv = np.linspace(-p.R, p.R, cfg.mpc_nu)

        blocks = []
        for b in range(nb):
            seg = ds[k + b*per: k + (b+1)*per]
            if len(seg) == 0:
                seg = np.full(per, ds[-1])
            elif len(seg) < per:
                seg = np.concatenate([seg, np.full(per - len(seg), seg[-1])])
            if forecast_bias:
                seg = np.clip(seg * (1 - forecast_bias), p.d_lo, p.d_hi)
            blocks.append(seg)

        trans = [_block_transition(p, cfg, blk, s_grid, g_grid, u_lv) for blk in blocks]

        # terminal cost: 0, or +inf outside the kernel when the constraint is on (B2)
        SS, GG = np.meshgrid(s_grid, g_grid, indexing="ij")
        J = np.zeros((cfg.mpc_ns, cfg.mpc_ng))
        if terminal_viability:
            J = np.where(margin(SS, GG, p) >= 0, 0.0, 1e9)

        # backward DP over the decision sequence
        policies = []
        for b in range(nb - 1, -1, -1):
            S_n, G_n, cost = trans[b]
            Q = cost + _interp_cost(J, s_grid, g_grid, S_n, G_n)
            policies.insert(0, u_lv[np.argmin(Q, axis=2)])
            J = Q.min(axis=2)

        ds_ = s_grid[1] - s_grid[0]; dg_ = g_grid[1] - g_grid[0]
        i = int(np.clip(round((s - s_grid[0]) / ds_), 0, cfg.mpc_ns - 1))
        j = int(np.clip(round((g - g_grid[0]) / dg_), 0, cfg.mpc_ng - 1))
        return float(policies[0][i, j])

    def pol(s, g, d, k, ds, p):
        t = k * p.dt
        if t >= state["t_next"] - 1e-12:          # re-solve at each block boundary
            state["plan"] = solve(s, g, k, ds, p)
            state["t_next"] = t + cfg.mpc_block
        return state["plan"]

    pol.reset = lambda: state.update(t_next=-1.0, plan=None)
    return pol


def run_mpc(pol, d_series, p, **kw):
    """MPC objects hold receding-horizon state; reset before each run."""
    pol.reset()
    return simulate(pol, d_series, p, **kw)

