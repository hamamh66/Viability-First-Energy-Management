"""Extensions used in the revision: battery/inverter power limit, abrupt drops,
robust and chance-constrained MPC, implementation uncertainty, sampled-data tightness."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.stats import norm
from vfem_core import (Params, Cfg, CFG, P, TOL, s_min, margin, admissible_u, simulate,
                       worst_case_series, ou_series, greedy, viable_bangbang,
                       viable_proportional, fixed_reserve, affine_cost, _interp_cost)

INF = float("inf")


@dataclass(frozen=True)
class PX(Params):
    P_b: float = INF          # kW  battery/inverter power limit (discharge and charge)


PXN = PX()                    # nominal, no power limit (identical to P)


def g_floor(p):
    """Lowest dispatch from which an instantaneous jump to d_hi is servable."""
    return max(p.d_hi - p.L_nc - p.P_b, 0.0) if np.isfinite(p.P_b) else 0.0


def margin_x(s, g, p):
    """Margin of the power-limited kernel; strongly negative below the dispatch floor."""
    m = s - s_min(g, p)
    gf = g_floor(p)
    return np.where(np.asarray(g) >= gf - 1e-12, m, m - 1e3 * (gf - np.asarray(g)))


def step_x(s, g, u, d, p):
    """Exact-ramp step with optional battery power limit. Returns s', g', shed, throughput."""
    u = admissible_u(u, g, p)
    g_new = min(max(g + u * p.dt, 0.0), p.G_max)
    g_avg = 0.5 * (g + g_new)
    shed = 0.0
    thr = 0.0
    net = g_avg - d                                  # >0 charge, <0 discharge
    if net < 0:
        need = -net
        deliver = min(need, p.P_b)
        shed += (need - deliver) * p.dt              # inverter-limited shortfall
        de = -deliver * p.dt / p.eta
        thr = deliver * p.dt
    else:
        de = min(net, p.P_b) * p.dt
    s_new = s + de
    if s_new < p.s_dod:
        short = (p.s_dod - s_new) * p.eta            # energy that could not be delivered
        shed += short
        thr -= short
        s_new = p.s_dod
    s_new = min(s_new, p.S_max)
    return s_new, g_new, shed, thr


def simulate_x(policy, d_series, p, s0=None, g0=None, cfg=CFG, record=False,
               delay=0, soc_bias=0.0, soc_noise=0.0, rng=None):
    """Plant simulation with power limit, actuation delay (samples) and SoC measurement error."""
    s = cfg.s0 if s0 is None else s0
    g = cfg.g0 if g0 is None else g0
    fuel = shed = on_time = thr = 0.0
    starts = 0
    prev_on = g > TOL
    queue = [0.0] * delay
    mus = []
    hist = {"s": [], "g": []}
    for k, d in enumerate(d_series):
        s_meas = s + soc_bias + (soc_noise * rng.normal() if soc_noise else 0.0)
        u_cmd = policy(s_meas, g, d, k, d_series, p)
        if delay:
            queue.append(u_cmd)
            u = queue.pop(0)
        else:
            u = u_cmd
        s, g, sh, th = step_x(s, g, u, d, p)
        shed += sh; thr += th; fuel += g * p.dt
        on = g > TOL
        on_time += p.dt if on else 0.0
        starts += int(on and not prev_on)
        prev_on = on
        mus.append(float(margin_x(s, g, p)))
        if record:
            hist["s"].append(s); hist["g"].append(g)
    out = dict(fuel=fuel, shed=shed, on_time=on_time, starts=starts, thr=thr,
               min_mu=min(mus), efc=thr / p.S_max)
    if record:
        out |= {k: np.array(v) for k, v in hist.items()}
        out["mu"] = np.array(mus)
    return out


# ---------------------------------------------------------------- policies with dispatch floor
def bb_x(mu_bar):
    def pol(s, g, d, k, ds, p):
        if margin_x(s, g, p) <= mu_bar:
            return +p.R
        gf = g_floor(p)
        if g - p.R * p.dt < gf:                      # do not ramp below the floor
            return float(np.clip((gf - g) / p.dt, -p.R, p.R))
        return -p.R
    return pol


def prop_x(mu_bar, kp=CFG.kp_prop):
    def pol(s, g, d, k, ds, p):
        u = kp * (mu_bar - float(margin_x(s, g, p)))
        gf = g_floor(p)
        return max(u, (gf - g) / p.dt) if np.isfinite(gf) else u
    return pol


def bb_state(mu0):
    """State-dependent sampled-data trigger: mu <= mu0 + 2(d_hi-g)_+ dt + R dt^2."""
    def pol(s, g, d, k, ds, p):
        thr = mu0 + 2 * max(p.d_hi - g, 0.0) * p.dt + p.R * p.dt**2
        return +p.R if margin(s, g, p) <= thr else -p.R
    return pol


def wrap(pol):
    return lambda s, g, d, k, ds, p: pol(s, g, d, k, ds, p)


# ---------------------------------------------------------------- disturbances
def step_drought(p, cfg=CFG, t0=2.5, t1=7.5, T=12.0):
    t = np.arange(0, T, p.dt)
    d = np.full_like(t, p.d_lo)
    d[(t >= t0) & (t < t1)] = p.d_hi
    return t, d


def recurring_droughts(p, T=24.0, blocks=((2.5, 5.5), (6.5, 9.5), (10.5, 13.5))):
    t = np.arange(0, T, p.dt)
    d = np.full_like(t, p.d_lo)
    for a, b in blocks:
        d[(t >= a) & (t < b)] = p.d_hi
    return t, d


def flicker_series(rng, p, T=24.0, dwell=(0.25, 2.0)):
    """Abrupt switching between surplus and worst-case deficit with random dwell times."""
    n = int(T / p.dt)
    d = np.empty(n)
    k = 0
    level = p.d_lo if rng.random() < 0.5 else p.d_hi
    while k < n:
        m = max(1, int(rng.uniform(*dwell) / p.dt))
        d[k:k + m] = level
        k += m
        level = p.d_hi if level == p.d_lo else p.d_lo
    return d


def collapse_series(rng, p, cfg=CFG):
    """OU deficit of Eq. (21) with abrupt renewable collapses: 1-3 instantaneous jumps from a
    surplus level to d_hi lasting 0.5-4 h, each preceded by 1 h of surplus at d_lo."""
    d = ou_series(rng, p, cfg)
    n = len(d)
    t = np.arange(n) * p.dt
    for _ in range(rng.integers(1, 4)):
        dur = rng.uniform(0.5, 4.0)
        t0 = rng.uniform(1.0, cfg.T_mc - dur)
        d[(t >= t0 - 1.0) & (t < t0)] = p.d_lo
        d[(t >= t0) & (t < t0 + dur)] = p.d_hi
    return d


# ---------------------------------------------------------------- MPC family
def _block_transition_x(p, cfg, d_block, s_grid, g_grid, u_levels):
    ns, ng, nu = len(s_grid), len(g_grid), len(u_levels)
    S = np.repeat(s_grid[:, None, None], ng, 1).repeat(nu, 2).astype(float)
    G = np.repeat(g_grid[None, :, None], ns, 0).repeat(nu, 2).astype(float)
    U = np.repeat(u_levels[None, None, :], ns, 0).repeat(ng, 1).astype(float)
    cost = np.zeros_like(S)
    for d in d_block:
        u = np.clip(U, np.where(G > TOL, -p.R, 0.0), np.where(G < p.G_max - TOL, p.R, 0.0))
        G_new = np.clip(G + u * p.dt, 0.0, p.G_max)
        ga = 0.5 * (G + G_new)
        net = ga - d
        need = np.maximum(-net, 0.0)
        deliver = np.minimum(need, p.P_b)
        shed = (need - deliver) * p.dt
        de = np.where(net < 0, -deliver * p.dt / p.eta, np.minimum(net, p.P_b) * p.dt)
        S_new = S + de
        shed += np.maximum(p.s_dod - S_new, 0.0)
        S_new = np.minimum(np.maximum(S_new, p.s_dod), p.S_max)
        cost += G_new * p.dt + cfg.mpc_shed_penalty * shed
        S, G = S_new, G_new
    return S, G, cost


def mpc_x(H, mode="nominal", bias=0.0, sigma_f=0.0, w=0.0, eps=0.05,
          terminal=False, persistence=False, cfg=CFG):
    """Economic MPC family on one DP solver.

    mode = 'nominal'  : plan on the forecast f
           'robust'   : plan on min(f + w, d_hi)  (min-max over the bounded error tube;
                        the shedding cost is monotone in d, so the upper edge is worst)
           'chance'   : plan on f + z_{1-eps} sigma_f (deterministic equivalent of
                        P(shed) <= eps per stage under additive Gaussian forecast error)
           'worst'    : plan on d_hi throughout (fully robust tube)
    Forecast f = (1-bias)*d_true over the horizon, or the persistence forecast
    f(t+tau) = d_true(t) when persistence=True.
    """
    state = {"t_next": -1.0, "plan": None}
    z = norm.ppf(1 - eps)

    def solve(s, g, k, ds, p):
        nb = max(1, int(round(H / cfg.mpc_block)))
        per = max(1, int(round(cfg.mpc_block / p.dt)))
        s_grid = np.linspace(p.s_dod, p.S_max, cfg.mpc_ns)
        g_grid = np.linspace(0.0, p.G_max, cfg.mpc_ng)
        u_lv = np.linspace(-p.R, p.R, cfg.mpc_nu)
        blocks = []
        for b in range(nb):
            if persistence:
                seg = np.full(per, ds[k])
            else:
                seg = ds[k + b * per: k + (b + 1) * per]
                if len(seg) == 0:
                    seg = np.full(per, ds[-1])
                elif len(seg) < per:
                    seg = np.concatenate([seg, np.full(per - len(seg), seg[-1])])
                seg = seg * (1 - bias)
            if mode == "robust":
                seg = seg + w
            elif mode == "chance":
                seg = seg + z * sigma_f
            elif mode == "worst":
                seg = np.full(per, p.d_hi)
            blocks.append(np.clip(seg, p.d_lo, p.d_hi))
        trans = [_block_transition_x(p, cfg, blk, s_grid, g_grid, u_lv) for blk in blocks]
        SS, GG = np.meshgrid(s_grid, g_grid, indexing="ij")
        J = np.zeros((cfg.mpc_ns, cfg.mpc_ng))
        if terminal:
            J = np.where(margin_x(SS, GG, p) >= 0, 0.0, 1e9)
        pols = []
        for b in range(nb - 1, -1, -1):
            S_n, G_n, cost = trans[b]
            Q = cost + _interp_cost(J, s_grid, g_grid, S_n, G_n)
            pols.insert(0, u_lv[np.argmin(Q, axis=2)])
            J = Q.min(axis=2)
        ds_ = s_grid[1] - s_grid[0]; dg_ = g_grid[1] - g_grid[0]
        i = int(np.clip(round((s - s_grid[0]) / ds_), 0, cfg.mpc_ns - 1))
        j = int(np.clip(round((g - g_grid[0]) / dg_), 0, cfg.mpc_ng - 1))
        return float(pols[0][i, j])

    def pol(s, g, d, k, ds, p):
        t = k * p.dt
        if t >= state["t_next"] - 1e-12:
            state["plan"] = solve(s, g, k, ds, p)
            state["t_next"] = t + cfg.mpc_block
        return state["plan"]

    pol.reset = lambda: state.update(t_next=-1.0, plan=None)
    return pol


def run(pol, d, p, **kw):
    if hasattr(pol, "reset"):
        pol.reset()
    return simulate_x(pol, d, p, **kw)


def nondominated(points):
    """Indices of non-dominated rows (minimisation in every column)."""
    pts = np.asarray(points, float)
    keep = []
    for i, a in enumerate(pts):
        dom = np.any(np.all(pts <= a + 1e-12, axis=1) & np.any(pts < a - 1e-12, axis=1))
        if not dom:
            keep.append(i)
    return keep
