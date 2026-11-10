import numpy as np


def _split(P):
    return P["B"], P["J"], P["inner"], P["norm"]


def _init_state(P, lam, n_prev=1):
    """Trạng thái đầu: x, x_prev[, x_prev2] đều trỏ tới bản sao của x0.
    An toàn vì các step không sửa mảng tại chỗ."""
    x0 = P["x0"].copy()
    s = dict(x=x0, x_prev=x0, lam=lam)
    if n_prev == 2:
        s["x_prev2"] = x0
    return s


def _adapt(base, c, num, dB):
    """lam_next = min(base, c * num / dB) nếu dB > 0, ngược lại = base."""
    return min(base, c * num / dB) if dB > 0 else base


# ---------------------------------------------------------------------------
# Our Alg.
# ---------------------------------------------------------------------------
def make_ours(P, p, f):
    B, J, inner, norm = _split(P)
    alpha_f, eta_f, delta_f, xi_f = p["alpha"], p["eta"], p["delta"], p["xi"]
    a_f, ak_f, b_f, bk_f = p["a"], p["a_k"], p["b"], p["b_k"]
    theta_max = p["theta"]
    a0, b0 = p["a"], p["b"]  # nếu p["a"], p["b"] là hằng số
    a_k, b_k = p["a_k"], p["b_k"]

    def step(s, k):
        x, xp, lam = s["x"], s["x_prev"], s["lam"]
        alpha = alpha_f(k)
        # Step 1: viscosity + (k lẻ) quán tính xen kẽ
        w = alpha * f(x) + (1 - alpha) * x
        if k % 2 == 1:
            d = norm(x - xp)
            theta_k = min(eta_f(k) / max(d * d, d), theta_max) if d != 0 else theta_max
            w = w + theta_k * (x - xp)
        # Step 2-3: backward + hiệu chỉnh Tseng
        Bw = B(w)
        y = J(w - lam * Bw, lam)
        diff = Bw - B(y)
        wy = w - y
        phi = wy - lam * diff
        # Step 4: bước nhảy tự thích nghi
        lam_next = _adapt(lam * delta_f(k) + xi_f(k), a0 + a_k(k), norm(wy), norm(diff))
        # Step 5: nới lỏng thích nghi
        nphi = inner(phi, phi)
        beta = (b0 + b_k(k)) * inner(wy, phi) / nphi if nphi > 0 else 0.0
        return dict(x=w - beta * phi, x_prev=x, lam=lam_next)

    return _init_state(P, p["lam1"]), step


# ---------------------------------------------------------------------------
# Alg1 — Bot
# ---------------------------------------------------------------------------
def make_alg1(P, p):
    B, J, inner, norm = _split(P)
    a, alpha = p["a"], p["inertia"]
    const = p["stepsize_mode"] == "constant"
    table = p["rho_mode"] == "table"
    # rho cố định khi không dùng bảng
    rho_fixed = 0.99 * 2.0 * (1 - alpha) ** 2 / ((1 + a) * (2 * alpha**2 - alpha + 1))

    def step(s, k):
        x, xp, lam = s["x"], s["x_prev"], s["lam"]
        z = x + alpha * (x - xp)
        Bz = B(z)
        y = J(z - lam * Bz, lam)
        diff = B(y) - Bz
        if const:
            lam_next, rho = lam, (1.0 / (1.0 + a) if table else rho_fixed)
        else:
            lam_next = _adapt(lam, a, norm(y - z), norm(diff))
            rho = 1.0 / (1.0 + a * lam / lam_next) if table else rho_fixed
        x_next = (1 - rho) * z + rho * (y - lam * diff)
        return dict(x=x_next, x_prev=x, lam=lam_next)

    lam0 = a / P["ell"] if const else p["lam1"]
    return _init_state(P, lam0), step


# ---------------------------------------------------------------------------
# Alg2 — Cholamjiak
# ---------------------------------------------------------------------------
def make_alg2(P, p):
    B, J, inner, norm = _split(P)
    alpha, theta, mu = p["alpha"], p["theta"], p["mu"]

    def step(s, k):
        x, xp, lam = s["x"], s["x_prev"], s["lam"]
        w = x + alpha * (x - xp)
        Bw = B(w)
        y = J(w - lam * Bw, lam)
        diff = Bw - B(y)
        x_next = (1 - theta) * w + theta * y + theta * lam * diff
        lam_next = _adapt(lam, mu, norm(w - y), norm(diff))
        return dict(x=x_next, x_prev=x, lam=lam_next)

    return _init_state(P, p["lam1"]), step


# ---------------------------------------------------------------------------
# Hai bước quán tính dùng cho Alg3 / Alg4
# ---------------------------------------------------------------------------
def _make_inertia(p, norm):
    eta_f, tau_f, muk_f = p["eta"], p["tau"], p["mu_k"]

    def inertia(x, xp, xpp, k):
        eta = eta_f(k)
        d1 = norm(x - xp)
        theta = min(eta / d1, tau_f(k)) if d1 != 0 else tau_f(k)
        d2 = norm(xp - xpp)
        mu_k = muk_f(k)
        delta = max(-eta / d2, -mu_k) if d2 != 0 else -mu_k
        return theta * (x - xp) + delta * (xp - xpp)

    return inertia


# ---------------------------------------------------------------------------
# Alg3 — Thuy & Ly, Algorithm 3.1, N = 1
# ---------------------------------------------------------------------------
def make_alg3(P, p, F):
    B, J, inner, norm = _split(P)
    inertia = _make_inertia(p, norm)
    alpha_f, rho_f, xi_f, mu = p["alpha"], p["rho"], p["xi"], p["mu"]

    def step(s, k):
        x, xp, xpp, lam = s["x"], s["x_prev"], s["x_prev2"], s["lam"]
        w = x + inertia(x, xp, xpp, k)
        Bw = B(w)
        y = J(w - lam * (Bw + alpha_f(k) * F(w)), lam)
        diff = B(y) - Bw
        x_next = y - lam * diff
        lam_next = _adapt(lam + rho_f(k), mu + xi_f(k), norm(y - w), norm(diff))
        return dict(x=x_next, x_prev=x, x_prev2=xp, lam=lam_next)

    return _init_state(P, p["lam1"], n_prev=2), step


# ---------------------------------------------------------------------------
# Alg4 — Thuy, Tung, Ly (COAM 2024), Algorithm 3.1, N = 1
# ---------------------------------------------------------------------------
def make_alg4(P, p, f):
    B, J, inner, norm = _split(P)
    inertia = _make_inertia(p, norm)
    alpha_f, mu = p["alpha"], p["mu"]

    def step(s, k):
        x, xp, xpp, lam = s["x"], s["x_prev"], s["x_prev2"], s["lam"]
        a_k = alpha_f(k)
        w = a_k * f(x) + (1 - a_k) * x + inertia(x, xp, xpp, k)
        Bw = B(w)
        y = J(w - lam * Bw, lam)
        diff = B(y) - Bw
        x_next = y - lam * diff
        lam_next = _adapt(lam, mu, norm(y - w), norm(diff))
        return dict(x=x_next, x_prev=x, x_prev2=xp, lam=lam_next)

    return _init_state(P, p["lam1"], n_prev=2), step