from pv_mfaps_plugin import build_plan


def test_plan_is_legal_and_deterministic():
    cal = {
        "A": [1.0, 1.2, 0.8, 1.4],
        "rho": [0.7, 0.8, 0.9, 0.75],
        "sigma": [0.1, 0.12, 0.08, 0.15],
        "n": [5, 8, 16, 32],
    }
    p1 = build_plan(cal, budget=0.5, k_grid=(0, 1, 2), alpha_grid=(0.25,))
    p2 = build_plan(cal, budget=0.5, k_grid=(0, 1, 2), alpha_grid=(0.25,))
    assert p1.k == p2.k
    assert p1.alpha == p2.alpha
    assert len(p1.k) == 4
    assert all(k == 0 or a == 0.25 for k, a in zip(p1.k, p1.alpha))
