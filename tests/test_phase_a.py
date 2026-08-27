"""Minimal invariants for the Phase-A substrate. Run: python tests/test_phase_a.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csc import estimators as est
from csc.model import ToyConfig, ToyModel
from csc.attack import delta_star_analytic, delta_star_pgd, input_evasion_gain


def test_welch_floor():
    assert abs(est.welch_floor(128, 32) - np.sqrt((128 - 32) / (32 * 127))) < 1e-9
    assert est.welch_floor(16, 32) == 0.0


def test_bound_respected():
    for (F, d) in [(64, 16), (128, 32), (256, 32), (512, 64)]:
        m = ToyModel(ToyConfig(F=F, d_eff=d, geometry="random", seed=1))
        assert m.coherence() >= m.welch_floor() - 1e-9


def test_delta_matches_prop1():
    """S == beta / (Gamma_r * ||a||) exactly (Proposition 1)."""
    m = ToyModel(ToyConfig(F=128, d_eff=32, geometry="random", seed=2))
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, w_b, U = m.plant_pair(rng)
        res = delta_star_analytic(m.W, w_b, U, beta=1.0, tau=0.0)
        g, lev = input_evasion_gain(m.W, w_b, U)
        if res.feasible:
            assert abs(res.delta_input - 1.0 / (g * lev)) < 1e-6


def test_sign_of_effect():
    """Median S decreases as load F/d rises (superposition erodes safety)."""
    meds = []
    for F in [32, 64, 128, 256]:
        m = ToyModel(ToyConfig(F=F, d_eff=32, geometry="random",
                               monitor_angle_deg=20.0, seed=3))
        rng = np.random.default_rng(7)
        S = [delta_star_analytic(m.W, *m.plant_pair(rng)[1:]).delta_input
             for _ in range(50)]
        S = np.array(S)
        meds.append(np.median(S[np.isfinite(S)]))
    assert all(meds[i] > meds[i + 1] for i in range(len(meds) - 1)), meds


def test_pgd_upper_bounds_analytic():
    m = ToyModel(ToyConfig(F=128, d_eff=32, geometry="random", seed=4))
    rng = np.random.default_rng(1)
    _, w_b, U = m.plant_pair(rng)
    an = delta_star_analytic(m.W, w_b, U).delta_input
    pg = delta_star_pgd(m.W, w_b, U, steps=1500, restarts=4)
    assert pg >= an - 1e-3           # PGD is an upper bound on the exact minimum
    assert pg / an < 2.0             # within a constant factor (X1 requirement)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
