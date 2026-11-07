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
Regimes: d_r small (the read is the bottleneck) vs d_r large (carriers become the
bottleneck under compression).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evasion import build_monitor_basis, nested_capture
from . import stats as st

__all__ = ["ToyCfg", "train", "measure", "sweep", "summarize", "self_check", "DEFAULT_REGIMES"]


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


DEFAULT_REGIMES = {"read_bottleneck": dict(dr=8), "carrier_bottleneck": dict(dr=48)}


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


def measure(m: ToyModel, cfg: ToyCfg, ranks=(1, 2, 4), n_eval=2000, n_mon=2000, seed=123) -> Dict:
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
    crand = np.random.default_rng(seed + 3).standard_normal(cfg.dr)
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
            Mcache[key] = (JJ, K)
        JJ, K = Mcache[key]
        gn2 = float(w0 @ JJ @ w0)
        cap = nested_capture(K, U.T @ JJ @ w0, gn2)
        pn2 = float(crand @ JJ @ crand)
        capp = nested_capture(K, U.T @ JJ @ crand, pn2)
        recs.append({"image": i, "m": abs(float(b[i])), "gn2": gn2, "cap": cap, "cap_pl": capp,
                     "correct": int((b[i] > 0) == (ye[i] == 1))})
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
            "accuracy": float(np.mean([r["correct"] for r in recs])),
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
                        cap = float(np.median([r["cap"][-1] for r in res["recs"]]))
                        print(f"  [{rname:18s}] {method:11s} seed={seed} keep={keep:.3f} "
                              f"loss={res['final_loss']:.4f} acc={res['accuracy']:.3f} "
                              f"F_eff={res['F_eff']:3d} cap@{ranks[-1]}={cap:.3f}", flush=True)
            R["regimes"][rname][method] = cells
    R["summary"] = summarize(R)
    return R


def summarize(R: Dict, B: int = 1000) -> Dict:
    S = {}
    for rname, meths in R["regimes"].items():
        S[rname] = {}
        for method, cells in meths.items():
            x, g, lm, ng, lT, A, Ap = [], [], [], [], [], [], []
            for c in cells:
                for rec in c["recs"]:
                    cap = float(np.clip(rec["cap"][-1], 1e-12, 1 - 1e-12))
                    capp = float(np.clip(rec["cap_pl"][-1], 1e-12, 1 - 1e-12))
                    # x = log of the ACTUAL carrier fraction (pool rounding differs from nominal keep)
                    x.append(math.log(c["carrier_dim"] / (c["carriers_full"] if "carriers_full" in c else c["carrier_dim_full"])))
                    g.append(c["seed"] * 100000 + rec["image"])
                    lm.append(math.log(rec["m"] + 1e-9)); ng.append(-0.5 * math.log(rec["gn2"] + 1e-30))
                    lT.append(-0.5 * math.log(1 - cap)); A.append(-math.log(cap)); Ap.append(-math.log(capp))
            x, g = np.array(x), np.array(g)
            lm, ng, lT, A, Ap = map(np.array, (lm, ng, lT, A, Ap))
            dec = st.decompose(x, {"log_oc": lm + ng + lT, "log_m": lm, "neg_log_g": ng, "log_T": lT},
                               g, B=B)
            keeps = sorted({c["keep"] for c in cells}, reverse=True)
            per_keep = []
            for k in keeps:
                ck = [c for c in cells if c["keep"] == k]
                per_keep.append({"keep": k, "carrier_dim": ck[0]["carrier_dim"],
                                 "F_eff": float(np.mean([c["F_eff"] for c in ck])),
                                 "accuracy": float(np.mean([c["accuracy"] for c in ck])),
                                 "cap_in": float(np.median([r["cap"][-1] for c in ck for r in c["recs"]])),
                                 "cap_pl": float(np.median([r["cap_pl"][-1] for c in ck for r in c["recs"]])),
                                 "cap_read": float(np.mean([c["cap_read"][-1] for c in ck])),
                                 "coh_mean": float(np.mean([c["coh_mean"] for c in ck])),
                                 "loss": float(np.mean([c["final_loss"] for c in ck]))})
            fe = np.array([p["F_eff"] for p in per_keep]); cd = np.array([p["carrier_dim"] for p in per_keep])
            alpha = float(np.polyfit(np.log(cd), np.log(np.maximum(fe, 1)), 1)[0]) if len(cd) > 1 else float("nan")
            S[rname][method] = {"decomposition": dec,
                                "A": st.paired_slope(x, A, g, B=B),
                                "A_placebo": st.paired_slope(x, Ap, g, B=B),
                                "A_minus_placebo": st.paired_slope_diff(x, A, Ap, g, B=B),
                                "alpha_F_eff": alpha, "per_keep": per_keep}
    return S


def format_summary(S: Dict, r: int) -> str:
    L = [f"X1b trained toy (monitor rank r={r}); beta = d log X / d log(carrier fraction) "
         "(beta>0: X falls under compression)"]
    for rname, meths in S.items():
        L.append(f"\n== regime: {rname}")
        for method, M in meths.items():
            d = M["decomposition"]
            L.append(f" [{method}]  alpha(F_eff vs carrier dim) = {M['alpha_F_eff']:+.2f}")
            L.append("   " + st.describe_slope("log OC", d["log_oc"], "OC falls", "OC rises"))
            L.append("   " + st.describe_slope("  log m (info)", d["log_m"], "margin falls", "margin rises"))
            L.append("   " + st.describe_slope("  -log||g|| (robust)", d["neg_log_g"], "less robust", "more robust"))
            L.append("   " + st.describe_slope("  log T (audit tax)", d["log_T"],
                                               "TAX FALLS (auditability worse)", "TAX RISES (auditability better)"))
            L.append("   " + st.describe_slope("A - A_placebo", M["A_minus_placebo"],
                                               "monitor sees MORE than geometry predicts",
                                               "monitor sees LESS than geometry predicts"))
            L.append("     keep  carr_dim  F_eff   acc   cap_in  cap_pl  cap_read  coh_mean  loss")
            for p in M["per_keep"]:
                L.append(f"    {p['keep']:.3f}  {p['carrier_dim']:6d}  {p['F_eff']:5.1f}  {p['accuracy']:.3f}  "
                         f"{p['cap_in']:.3f}   {p['cap_pl']:.3f}   {p['cap_read']:.3f}    {p['coh_mean']:.3f}  {p['loss']:.4f}")
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
