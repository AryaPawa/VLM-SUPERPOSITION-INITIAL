"""
csc_vlm.toy_trained  --  X1b: a TRAINED toy in which superposition emerges
=========================================================================

Why X1b exists (first-principles review, 2026-11-07)
----------------------------------------------------
The X1 toy (toy.py) validates our estimators but assumes its own result.
``reads_packed`` defines compression as "more co-active features", and
``compress_prune`` removes hidden dimensions, not tokens. It cannot tell us which
way the auditability effect goes. There are two competing forces:

  coverage      fewer carriers -> lower-rank pull-back J -> a fixed-rank monitor
                captures MORE of any read direction (E[cap] ~ r / d_eff)
                -> the obfuscation tax T = 1/Gamma RISES (auditability better);
  interference  fewer carriers must pack the same features -> more overlap ->
                the behaviour gradient spreads outside the monitor
                -> T FALLS (auditability worse).

X1b trains small models END TO END at each budget and lets the geometry emerge.
That is the review's X1 specification: "train a small model; F, N, d known;
sweep N".

Model (linear up to the output ReLU, so first-order quantities are EXACT)
  z in R^F (feature 0 = behaviour, present in half the images and seen by every
  patch; the others are local to one patch and sparse)
  token_t   = E_t [z_0 ; z_{S_t}]               (N0 patch tokens, width d)
  carriers  = pool(tokens)  or  drop(tokens)    (k = keep * N0)
  h         = sum_j R_{pos(j)} c_j              (read position, width d_r)
  out       = ReLU(W_o h + b_o)                 (reconstruct all of z)
Training per budget = the "recovery finetune" of the plan: every budget gets its
best-trained model.

Measured exactly (no estimation error in the gradients):
  b   = W_o[0] h + b_o[0] - tau,    m = |b|
  g_b = J^T w_0,   G_U = U^T J      (J = dh/dcarriers, per example for drop)
  cap(r), T(r), A(r), OC, a placebo read direction, read-space cap,
  F_eff (features reconstructed with R^2 >= 0.5), read coherence vs Welch floor.

v2.1 (alternate-analysis review, 2026-11-07)
  * Regimes: carrier_bottleneck (d_r = 48: carriers become the bottleneck under
    compression) and subtle_behaviour (same, but a RARE behaviour seen by fewer
    patches -- less capacity, more superposed, closer to a VLM's minor feature).
    read_bottleneck (d_r = 8) was uninformative and is kept in LEGACY_REGIMES only.
  * Nulls: the random-read placebo is averaged over n_placebo directions, plus a
    MATCHED placebo (same monitored component as w_0 at rank r_max, random
    unmonitored part of the same norm): equal read-space capture at every r, so
    no ceiling/floor asymmetry. The toy has no bypass channel (b depends on the
    carriers only through h), so g_b = J^T w_0 exactly.
  * Full-capacity (Mahalanobis-optimal) read detector: cost = |b| / sqrt(w_0^T
    Sigma w_0) (Sigma = read covariance of the monitor set).
  * Incidental detection: does the canonical rank-1 probe (class-mean axis,
    midpoint threshold) flip on the plain attack delta = -1.25 b g/||g||^2? Exact
    here (h is linear in the carriers).
  * Every monitor rank in ``ranks`` is summarised (r = 1 is the canonical probe),
    and every slope is also computed PER SEED: the seed (a separately trained
    model) is the real replication unit, so sign agreement across seeds is
    reported next to the image-level CI, which ignores training variance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evasion import build_monitor_basis, nested_capture
from . import stats as st

__all__ = ["ToyCfg", "train", "measure", "sweep", "summarize", "format_summary", "self_check",
           "DEFAULT_REGIMES", "LEGACY_REGIMES", "nested_capture_many"]


@dataclass
class ToyCfg:
    N0: int = 36           # patch tokens (6x6 grid)
    Fp: int = 4            # local features per patch -> F = 1 + N0*Fp
    d: int = 4             # token width
    dr: int = 16           # read width
    sparsity: float = 0.06
    p_behaviour: float = 0.5
    q_vis: float = 0.5     # P(behaviour visible in a given patch | present) -- object covers part of the image
    steps: int = 1500
    batch: int = 512
    lr: float = 3e-3
    seed: int = 0

    @property
    def F(self) -> int:
        return 1 + self.N0 * self.Fp


DEFAULT_REGIMES = {"carrier_bottleneck": dict(dr=48),
                   "subtle_behaviour": dict(dr=48, p_behaviour=0.1, q_vis=0.3)}
LEGACY_REGIMES = {"read_bottleneck": dict(dr=8)}


def nested_capture_many(K: np.ndarray, Z: np.ndarray, gn2: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Vectorised ``nested_capture`` for one K and many z: Z [n, R], gn2 [n] -> [n, R]."""
    Z = np.atleast_2d(Z); gn2 = np.asarray(gn2, float).reshape(-1)
    R = Z.shape[1]
    out = np.zeros((Z.shape[0], R))
    for r in range(1, R + 1):
        Kr = 0.5 * (K[:r, :r] + K[:r, :r].T)
        w, V = np.linalg.eigh(Kr)
        keep = w > tol * max(w.max(), 1e-12)
        zr = Z[:, :r] @ V[:, keep]
        out[:, r - 1] = np.sum(zr ** 2 / w[keep], axis=1) / np.maximum(gn2, 1e-300)
    out[gn2 <= 0] = 0.0
    return np.clip(out, 0.0, 1.0)


def sample_z(cfg: ToyCfg, n: int, rng) -> np.ndarray:
    Z = (rng.random((n, cfg.F)) < cfg.sparsity) * rng.random((n, cfg.F))
    on = rng.random(n) < cfg.p_behaviour
    Z[:, 0] = on * (0.3 + 0.7 * rng.random(n))
    return Z


def patch_inputs(cfg: ToyCfg, Z: np.ndarray, rng) -> np.ndarray:
    """X [n, N0, 1+Fp]: patch t sees its own local features, plus the behaviour
    feature if the 'object' covers it (prob q_vis per patch). Dropping patches can
    therefore delete behaviour evidence, and pooling mixes it with clutter."""
    n = Z.shape[0]
    loc = Z[:, 1:].reshape(n, cfg.N0, cfg.Fp)
    vis = (rng.random((n, cfg.N0)) < cfg.q_vis).astype(float)
    return np.concatenate([(Z[:, :1] * vis)[..., None], loc], axis=2)


def pool_matrix(N0: int, keep: float) -> np.ndarray:
    """2-D average pooling on a sqrt(N0) x sqrt(N0) grid (1-D if N0 not square)."""
    s = int(round(math.sqrt(N0)))
    if s * s == N0:
        from .compression import pool_groups_2d
        groups = pool_groups_2d(s, s, keep)
    else:
        from .compression import pool_groups_1d
        groups = pool_groups_1d(N0, keep)
    P = np.zeros((len(groups), N0))
    for j, g in enumerate(groups):
        P[j, g] = 1.0 / len(g)
    return P


def select_mask(method: str, keep: float, tok: np.ndarray, rng) -> np.ndarray:
    """[n, N0] 0/1 keep mask for drop methods (positions preserved, dropped -> zero)."""
    n, N0, _ = tok.shape
    k = max(1, int(round(keep * N0)))
    M = np.zeros((n, N0))
    if method == "drop_random":
        idx = np.argsort(rng.random((n, N0)), axis=1)[:, :k]
    elif method == "drop_norm":
        idx = np.argsort(-np.linalg.norm(tok, axis=2), axis=1)[:, :k]
    else:
        raise ValueError(method)
    np.put_along_axis(M, idx, 1.0, axis=1)
    return M


class ToyModel:
    def __init__(self, cfg: ToyCfg, method: str, keep: float, rng):
        self.cfg, self.method, self.keep = cfg, method, float(keep)
        F, d, dr, N0 = cfg.F, cfg.d, cfg.dr, cfg.N0
        self.P = pool_matrix(N0, keep) if method == "pool" else None
        k_slots = self.P.shape[0] if self.P is not None else N0
        self.E = rng.normal(0, 1 / math.sqrt(1 + cfg.Fp), (N0, d, 1 + cfg.Fp))
        self.R = rng.normal(0, 1 / math.sqrt(k_slots * d), (k_slots, dr, d))
        self.W = rng.normal(0, 1 / math.sqrt(dr), (F, dr))
        self.b = np.zeros(F)

    def params(self):
        return [self.E, self.R, self.W, self.b]

    def forward(self, X, rng, cache=False):
        tok = np.matmul(X.transpose(1, 0, 2), self.E.transpose(0, 2, 1)).transpose(1, 0, 2)
        if self.method == "pool":
            C = np.einsum("ntd,jt->njd", tok, self.P, optimize=True)
            M = None
        elif self.keep >= 1.0 or self.method == "none":
            C, M = tok, np.ones(tok.shape[:2])
        else:
            M = select_mask(self.method, self.keep, tok, rng)
            C = tok * M[..., None]
        n_ = C.shape[0]
        h = C.reshape(n_, -1) @ self.R.transpose(0, 2, 1).reshape(-1, self.R.shape[1])
        o = h @ self.W.T + self.b
        if cache:
            return o, dict(X=X, tok=tok, C=C, M=M, h=h, o=o)
        return o, h, M

    def loss_grad(self, Z, X, rng):
        o, c = self.forward(X, rng, cache=True)
        out = np.maximum(o, 0)
        n = Z.shape[0]
        L = float(((out - Z) ** 2).sum(1).mean())
        do = 2 * (out - Z) / n * (o > 0)
        dW = do.T @ c["h"]; db = do.sum(0)
        dh = do @ self.W
        k_, dr_, d_ = self.R.shape
        Cf = c["C"].reshape(n, -1)
        dR = (Cf.T @ dh).reshape(k_, d_, dr_).transpose(0, 2, 1)
        dC = (dh @ self.R.transpose(0, 2, 1).reshape(-1, dr_).T).reshape(n, k_, d_)
        dtok = np.einsum("njd,jt->ntd", dC, self.P, optimize=True) if self.method == "pool" else (
            dC * c["M"][..., None] if c["M"] is not None else dC)
        dE = np.matmul(dtok.transpose(1, 2, 0), c["X"].transpose(1, 0, 2))
        return L, [dE, dR, dW, db]


def train(cfg: ToyCfg, method: str, keep: float, verbose: bool = False) -> Tuple[ToyModel, List[float]]:
    rng = np.random.default_rng([cfg.seed, int(keep * 1e4), sum(map(ord, method))])
    m = ToyModel(cfg, method, keep, rng)
    ps = m.params()
    mom = [np.zeros_like(p) for p in ps]; vel = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, cfg.steps + 1):
        Z = sample_z(cfg, cfg.batch, rng)
        L, gs = m.loss_grad(Z, patch_inputs(cfg, Z, rng), rng)
        lr = cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))
        for p, g, mo, ve in zip(ps, gs, mom, vel):
            mo *= b1; mo += (1 - b1) * g
            ve *= b2; ve += (1 - b2) * g * g
            p -= lr * (mo / (1 - b1 ** t)) / (np.sqrt(ve / (1 - b2 ** t)) + eps)
        if t % max(1, cfg.steps // 10) == 0:
            hist.append(L)
            if verbose:
                print(f"    step {t} loss {L:.4f}")
    return m, hist


def _J_metric(m: ToyModel, M_row: Optional[np.ndarray]) -> np.ndarray:
    """J J^T in read space (dr x dr) for one example: sum over carriers present."""
    if m.method == "pool" or M_row is None:
        return np.einsum("jrd,jsd->rs", m.R, m.R)
    Rk = m.R[M_row > 0]
    return np.einsum("jrd,jsd->rs", Rk, Rk)


def feature_dirs(m: ToyModel, M_row: Optional[np.ndarray]) -> np.ndarray:
    """[dr, F] read direction of each latent feature (dh/dz) for one example."""
    cfg = m.cfg
    V = np.zeros((cfg.dr, cfg.F))
    if m.method == "pool":
        A = np.einsum("jt,tdf->jtdf", m.P, m.E)          # carrier j <- patch t features
        for j in range(m.P.shape[0]):
            RA = np.einsum("rd,tdf->trf", m.R[j], A[j])
            V[:, 0] += RA[:, :, 0].sum(0)
            for t in range(cfg.N0):
                V[:, 1 + t * cfg.Fp: 1 + (t + 1) * cfg.Fp] += RA[t, :, 1:]
    else:
        for t in range(cfg.N0):
            if M_row is not None and M_row[t] == 0:
                continue
            RA = m.R[t] @ m.E[t]
            V[:, 0] += RA[:, 0]
            V[:, 1 + t * cfg.Fp: 1 + (t + 1) * cfg.Fp] += RA[:, 1:]
    return V


def welch(F: int, d: int) -> float:
    return math.sqrt((F - d) / (d * (F - 1))) if F > d else 0.0


def measure(m: ToyModel, cfg: ToyCfg, ranks=(1, 2, 4), n_eval=2000, n_mon=2000, seed=123,
            n_placebo: int = 16, n_match: int = 8) -> Dict:
    rng = np.random.default_rng(seed)               # SAME eval inputs for every budget (paired)
    Zm = sample_z(cfg, n_mon, rng); Ze = sample_z(cfg, n_eval, rng)
    om, hm, _ = m.forward(patch_inputs(cfg, Zm, np.random.default_rng(seed + 11)),
                          np.random.default_rng(seed + 1))
    oe, he, Me = m.forward(patch_inputs(cfg, Ze, np.random.default_rng(seed + 12)),
                           np.random.default_rng(seed + 2))
    ym, ye = (Zm[:, 0] > 0).astype(int), (Ze[:, 0] > 0).astype(int)
    qs = np.quantile(om[:, 0], np.linspace(0.02, 0.98, 97))
    tau = float(qs[np.argmax([np.mean((om[:, 0] > q) == (ym == 1)) for q in qs])])
    b = oe[:, 0] - tau
    rmax = max(ranks)
    U = build_monitor_basis(hm, ym, rmax)
    w0 = m.W[0]
    # nulls: n_placebo random read directions; n_match matched placebos (same U_rmax component
    # as w0, random unmonitored component of the same norm -> same read-space cap at all r<=rmax)
    Cp = np.random.default_rng(seed + 3).standard_normal((n_placebo, cfg.dr))
    pc = U @ (U.T @ w0)
    Wm = np.random.default_rng(seed + 4).standard_normal((n_match, cfg.dr))
    Wm = Wm - (Wm @ U) @ U.T
    Wm = Wm / (np.linalg.norm(Wm, axis=1, keepdims=True) + 1e-30) * np.linalg.norm(w0 - pc)
    Cm = pc[None, :] + Wm
    # canonical rank-1 probe (class-mean axis u0, midpoint threshold) and the full-capacity cost
    u0 = U[:, 0]
    mu1, mu0 = hm[ym == 1].mean(0), hm[ym == 0].mean(0)
    tau_p = float(u0 @ (0.5 * (mu1 + mu0)))
    gap = float(np.linalg.norm(mu1 - mu0)) + 1e-12
    sd_b = float(np.std(hm @ w0, ddof=1))           # sqrt(w0^T Sigma w0)
    s_probe = he @ u0
    out = np.maximum(oe, 0)
    r2 = 1 - ((Ze - out) ** 2).sum(0) / (((Ze - Ze.mean(0)) ** 2).sum(0) + 1e-12)
    F_eff = int((r2 >= 0.5).sum())
    recs = []
    Mcache = {}
    for i in range(n_eval):
        Mi = None if Me is None else Me[i]
        key = None if Mi is None else Mi.tobytes()
        if key not in Mcache:
            JJ = _J_metric(m, Mi)
            K = U.T @ JJ @ U
            gn2 = float(w0 @ JJ @ w0)
            cap = nested_capture(K, U.T @ JJ @ w0, gn2)
            pl = nested_capture_many(K, Cp @ JJ @ U, np.einsum("pi,ij,pj->p", Cp, JJ, Cp))
            mt = nested_capture_many(K, Cm @ JJ @ U, np.einsum("pi,ij,pj->p", Cm, JJ, Cm))
            lg = lambda c: -np.log(np.clip(c, 1e-12, 1.0))
            Mcache[key] = (gn2, cap, pl.mean(0), lg(pl).mean(0), mt.mean(0), lg(mt).mean(0),
                           float(u0 @ JJ @ w0))
        gn2, cap, cpl, Apl, cmt, Amt, uJw = Mcache[key]
        ds = -1.25 * float(b[i]) * uJw / max(gn2, 1e-30)    # exact probe shift of the plain attack
        recs.append({"image": i, "m": abs(float(b[i])), "gn2": gn2, "cap": cap,
                     "cap_pl": cpl, "A_pl": Apl, "cap_match": cmt, "A_match": Amt,
                     "correct": int((b[i] > 0) == (ye[i] == 1)),
                     "det": bool((s_probe[i] > tau_p) != (s_probe[i] + ds > tau_p)),
                     "shift_rel": abs(ds) / gap})
    cap_read = [float(np.sum((U[:, :r].T @ w0) ** 2) / (w0 @ w0)) for r in ranks]
    Vs = [feature_dirs(m, None if Me is None else Me[i]) for i in range(min(32, n_eval))]
    coh = []
    for V in Vs:
        nrm = np.linalg.norm(V, axis=0)
        live = nrm > 1e-3 * nrm.max()
        Vn = V[:, live] / nrm[live]
        Gm = np.abs(Vn.T @ Vn); np.fill_diagonal(Gm, 0)
        coh.append((Gm.max(), Gm.sum() / max(1, live.sum() * (live.sum() - 1)), int(live.sum())))
    k = m.P.shape[0] if m.P is not None else int(round(m.keep * cfg.N0)) if m.keep < 1 else cfg.N0
    return {"recs": recs, "ranks": list(ranks), "cap_read": cap_read, "F_eff": F_eff,
            "accuracy": float(np.mean([r["correct"] for r in recs])), "sd_b": sd_b,
            "probe_accuracy": float(np.mean((s_probe > tau_p) == (ye == 1))),
            "coh_max": float(np.median([c[0] for c in coh])), "coh_mean": float(np.median([c[1] for c in coh])),
            "n_live": float(np.median([c[2] for c in coh])), "welch": welch(cfg.F, cfg.dr),
            "carriers": k, "carrier_dim": k * cfg.d, "carrier_dim_full": cfg.N0 * cfg.d}


def sweep(regimes=None, methods=("pool", "drop_random", "drop_norm"), keeps=(1.0, 0.5, 0.25, 0.11),
          seeds=(0, 1, 2), ranks=(1, 2, 4), steps: int = 1500, verbose: bool = True,
          base: Optional[ToyCfg] = None) -> Dict:
    regimes = regimes or DEFAULT_REGIMES
    base = base or ToyCfg()
    R = {"config": base.__dict__, "regimes": {}, "keeps": list(keeps), "ranks": list(ranks)}
    for rname, over in regimes.items():
        R["regimes"][rname] = {}
        for method in methods:
            cells = []
            for seed in seeds:
                cfg = replace(base, seed=seed, steps=steps, **over)
                for keep in keeps:
                    model, hist = train(cfg, method, keep)
                    res = measure(model, cfg, ranks=ranks)
                    res.update({"seed": seed, "keep": keep, "final_loss": hist[-1] if hist else float("nan")})
                    cells.append(res)
                    if verbose:
                        cap = float(np.median([r["cap"][0] for r in res["recs"]]))
                        print(f"  [{rname:18s}] {method:11s} seed={seed} keep={keep:.3f} "
                              f"loss={res['final_loss']:.4f} acc={res['accuracy']:.3f} "
                              f"F_eff={res['F_eff']:3d} cap@1={cap:.3f}", flush=True)
            R["regimes"][rname][method] = cells
    R["summary"] = summarize(R)
    return R


_TERMS = ("log_oc", "log_m", "neg_log_g", "log_T")
_EXTRA = ("A", "A_match", "A_placebo", "A_minus_match", "A_minus_placebo", "log_full")


def _cols(cells, r: int):
    """Per-row arrays at monitor rank r. x = log of the ACTUAL carrier fraction."""
    i = r - 1
    rows = []
    for c in cells:
        lx = math.log(c["carrier_dim"] / c["carrier_dim_full"])
        lsd = math.log(c["sd_b"] + 1e-30)
        for rec in c["recs"]:
            cap = float(np.clip(rec["cap"][i], 1e-12, 1 - 1e-12))
            lm = math.log(rec["m"] + 1e-9)
            ng = -0.5 * math.log(rec["gn2"] + 1e-30)
            lT = -0.5 * math.log(1 - cap)
            A = -math.log(cap)
            rows.append((lx, c["seed"] * 100000 + rec["image"], c["seed"], lm + ng + lT, lm, ng, lT, A,
                         float(rec["A_match"][i]), float(rec["A_pl"][i]), lm - lsd))
    a = np.array(rows, float)
    d = {"x": a[:, 0], "g": a[:, 1], "seed": a[:, 2], "log_oc": a[:, 3], "log_m": a[:, 4],
         "neg_log_g": a[:, 5], "log_T": a[:, 6], "A": a[:, 7], "A_match": a[:, 8], "A_placebo": a[:, 9],
         "log_full": a[:, 10]}
    d["A_minus_match"] = d["A"] - d["A_match"]
    d["A_minus_placebo"] = d["A"] - d["A_placebo"]
    return d


def _per_seed(d, keys):
    """Point slope per seed (each seed = one separately trained set of models) and how
    many seeds agree in sign with the pooled slope."""
    out = {}
    seeds = sorted(set(d["seed"].tolist()))
    for k in keys:
        v = []
        for s_ in seeds:
            sel = d["seed"] == s_
            v.append(st.within_slope(d["x"][sel], d[k][sel], d["g"][sel]))
        pooled = st.within_slope(d["x"], d[k], d["g"])
        v = np.array(v, float)
        out[k] = {"per_seed": v.tolist(), "pooled": pooled,
                  "n_agree": int(np.sum(np.sign(v) == np.sign(pooled))), "n_seeds": len(seeds),
                  "seed_sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}
    return out


def summarize(R: Dict, B: int = 1000, ranks=None) -> Dict:
    """S[r][regime][method] for every monitor rank r (r = 1 = canonical probe)."""
    ranks = list(ranks or R["ranks"])
    S: Dict = {}
    for r in ranks:
        S[r] = {}
        for rname, meths in R["regimes"].items():
            S[r][rname] = {}
            for method, cells in meths.items():
                d = _cols(cells, r)
                x, g = d["x"], d["g"]
                dec = st.decompose(x, {k: d[k] for k in _TERMS}, g, B=B)
                ex = {k: st.paired_slope(x, d[k], g, B=B) for k in _EXTRA}
                keeps = sorted({c["keep"] for c in cells}, reverse=True)
                per_keep = []
                for k in keeps:
                    ck = [c for c in cells if c["keep"] == k]
                    rr = [rec for c in ck for rec in c["recs"]]
                    per_keep.append({"keep": k, "carrier_dim": ck[0]["carrier_dim"],
                                     "F_eff": float(np.mean([c["F_eff"] for c in ck])),
                                     "accuracy": float(np.mean([c["accuracy"] for c in ck])),
                                     "probe_accuracy": float(np.mean([c["probe_accuracy"] for c in ck])),
                                     "cap_in": float(np.median([q["cap"][r - 1] for q in rr])),
                                     "cap_match": float(np.median([q["cap_match"][r - 1] for q in rr])),
                                     "cap_pl": float(np.median([q["cap_pl"][r - 1] for q in rr])),
                                     "cap_read": float(np.mean([c["cap_read"][ranks.index(r)] for c in ck]))
                                     if r in R["ranks"] else float("nan"),
                                     "T": float(np.median([1 / math.sqrt(1 - min(q["cap"][r - 1], 1 - 1e-12))
                                                           for q in rr])),
                                     "full_cost": float(np.median([q["m"] / (c["sd_b"] + 1e-30)
                                                                   for c in ck for q in c["recs"]])),
                                     "incidental_det": float(np.mean([q["det"] for q in rr])),
                                     "shift_rel": float(np.median([q["shift_rel"] for q in rr])),
                                     "coh_mean": float(np.mean([c["coh_mean"] for c in ck])),
                                     "loss": float(np.mean([c["final_loss"] for c in ck]))})
                fe = np.array([p["F_eff"] for p in per_keep]); cd = np.array([p["carrier_dim"] for p in per_keep])
                alpha = float(np.polyfit(np.log(cd), np.log(np.maximum(fe, 1)), 1)[0]) if len(cd) > 1 else float("nan")
                S[r][rname][method] = {"decomposition": dec, **ex,
                                       "per_seed": _per_seed(d, ("log_oc", "log_T", "A_minus_match", "log_full")),
                                       "alpha_F_eff": alpha, "per_keep": per_keep}
    return S


def format_summary(S: Dict, r: int) -> str:
    S = S[r]
    L = [f"X1b trained toy -- monitor rank r={r}" + ("  (CANONICAL rank-1 probe)" if r == 1 else "")
         + "\n  beta = d log X / d log(carrier fraction); beta>0: X falls under compression"
         + "\n  [seeds a/n] = seeds (separately trained models) whose own slope has the pooled sign"]
    for rname, meths in S.items():
        L.append(f"\n== regime: {rname}")
        for method, M in meths.items():
            d, ps = M["decomposition"], M["per_seed"]
            sa = lambda k: f"  [seeds {ps[k]['n_agree']}/{ps[k]['n_seeds']}]"
            L.append(f" [{method}]  alpha(F_eff vs carrier dim) = {M['alpha_F_eff']:+.2f}")
            L.append("   " + st.describe_slope("log OC", d["log_oc"], "OC falls", "OC rises") + sa("log_oc"))
            L.append("   " + st.describe_slope("  log m (info)", d["log_m"], "margin falls", "margin rises"))
            L.append("   " + st.describe_slope("  -log||g|| (robust)", d["neg_log_g"], "less robust", "more robust"))
            L.append("   " + st.describe_slope("  log T (audit tax)", d["log_T"],
                                               "TAX FALLS (auditability worse)", "TAX RISES (auditability better)")
                     + sa("log_T"))
            L.append("   " + st.describe_slope("A - A_matched", M["A_minus_match"],
                                               "monitor sees MORE than matched null",
                                               "monitor sees LESS than matched null") + sa("A_minus_match"))
            L.append("   " + st.describe_slope("A - A_placebo(avg)", M["A_minus_placebo"],
                                               "monitor sees MORE than random reads",
                                               "monitor sees LESS than random reads"))
            L.append("   " + st.describe_slope("full-capacity cost", M["log_full"],
                                               "falls (optimal detector loses ground)", "rises") + sa("log_full"))
            L.append("     keep  carr_dim  F_eff  acc   p_acc  cap_in  cap_mt  cap_pl  cap_read   T     "
                     "full   incid  coh_mean  loss")
            for p in M["per_keep"]:
                L.append(f"    {p['keep']:.3f}  {p['carrier_dim']:6d}  {p['F_eff']:5.1f}  {p['accuracy']:.3f} "
                         f"{p['probe_accuracy']:.3f}  {p['cap_in']:.3f}   {p['cap_match']:.3f}   {p['cap_pl']:.3f}"
                         f"   {p['cap_read']:.3f}   {p['T']:.3f}  {p['full_cost']:.3f}  {p['incidental_det']:.3f}"
                         f"   {p['coh_mean']:.3f}   {p['loss']:.4f}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# self-check: manual backprop == finite differences; closed-form OC == min-norm solve
# --------------------------------------------------------------------------- #
def self_check(verbose: bool = False) -> bool:
    cfg = ToyCfg(N0=4, Fp=3, d=3, dr=5, steps=60, batch=256, seed=0)
    ok = True
    for method in ("pool", "drop_random", "drop_norm"):
        rng = np.random.default_rng(0)
        m = ToyModel(cfg, method, 0.5, rng)
        m.b[:] = rng.normal(0, 0.1, m.b.shape)       # move off the ReLU kink at o = 0 for the FD check
        Z = sample_z(cfg, 64, np.random.default_rng(1)); X = patch_inputs(cfg, Z, np.random.default_rng(2))
        mask_rng_seed = 7
        L, gs = m.loss_grad(Z, X, np.random.default_rng(mask_rng_seed))
        for pi, p in enumerate(m.params()):
            idx = tuple(np.random.default_rng(pi).integers(0, s) for s in p.shape)
            old = p[idx]; e = 1e-6
            p[idx] = old + e; Lp, _ = m.loss_grad(Z, X, np.random.default_rng(mask_rng_seed))
            p[idx] = old - e; Lm, _ = m.loss_grad(Z, X, np.random.default_rng(mask_rng_seed))
            p[idx] = old
            fd = (Lp - Lm) / (2 * e)
            if abs(fd - gs[pi][idx]) > 1e-5 * (1 + abs(fd)):
                ok = False
                print(f"[x1b self-check] grad mismatch {method} param{pi}: fd={fd} an={gs[pi][idx]}")
        model, hist = train(replace(cfg, steps=200), method, 0.5)
        if not hist[-1] < hist[0]:
            ok = False
            print(f"[x1b self-check] loss did not decrease for {method}: {hist[0]} -> {hist[-1]}")
        res = measure(model, cfg, ranks=(1, 2), n_eval=50, n_mon=200)
        rec = res["recs"][0]
        # min-norm solve: min ||delta|| s.t. g.delta = -b, G delta = 0  -> norm = m / sqrt(gn2 (1-cap))
        oc = rec["m"] / math.sqrt(rec["gn2"] * (1 - rec["cap"][-1]))
        if oc <= 0 or not np.isfinite(oc):
            ok = False
            print(f"[x1b self-check] non-finite OC for {method}")
    # plain attack in an explicit carrier Jacobian: b -> -0.25 b, probe shift = formula
    rng = np.random.default_rng(5)
    m = ToyModel(cfg, "drop_random", 0.5, rng)
    Mi = np.array([1.0, 0.0, 1.0, 1.0])
    Jx = np.concatenate([m.R[j] * Mi[j] for j in range(cfg.N0)], axis=1)     # [dr, N0*d]
    if np.abs(Jx @ Jx.T - _J_metric(m, Mi)).max() > 1e-10:
        ok = False
        print("[x1b self-check] explicit J J^T != _J_metric")
    w0 = m.W[0]; u0 = rng.standard_normal(cfg.dr); bval = 0.9
    gvec = Jx.T @ w0; gn2 = float(gvec @ gvec)
    delta = -1.25 * bval * gvec / gn2
    if abs((bval + w0 @ (Jx @ delta)) - (-0.25 * bval)) > 1e-10 or \
            abs(u0 @ (Jx @ delta) - (-1.25 * bval * float(u0 @ _J_metric(m, Mi) @ w0) / gn2)) > 1e-10:
        ok = False
        print("[x1b self-check] plain-attack algebra mismatch")
    # vectorised nested capture == scalar version
    K = rng.standard_normal((3, 3)); K = K @ K.T
    Zz = rng.standard_normal((5, 3)); gg = np.sum(Zz ** 2, 1) * 3
    if np.abs(nested_capture_many(K, Zz, gg) - np.stack([nested_capture(K, z, g_) for z, g_ in zip(Zz, gg)])).max() > 1e-12:
        ok = False
        print("[x1b self-check] nested_capture_many != nested_capture")
    # explicit min-norm check on a random linear system
    rng = np.random.default_rng(3)
    J = rng.standard_normal((6, 20)); w = rng.standard_normal(6); U = np.linalg.qr(rng.standard_normal((6, 2)))[0]
    g = J.T @ w; G = U.T @ J; bval = 0.7
    A_ = np.vstack([g, G]); rhs = np.r_[-bval, 0, 0]
    delta = np.linalg.lstsq(A_, rhs, rcond=None)[0]
    cap = nested_capture(G @ G.T, G @ g, g @ g)[-1]
    if abs(np.linalg.norm(delta) - bval / math.sqrt((g @ g) * (1 - cap))) > 1e-8:
        ok = False
        print("[x1b self-check] closed-form OC != min-norm solve")
    if verbose:
        print(f"[x1b self-check] {'PASS' if ok else 'FAIL'}")
    return ok