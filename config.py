"""Tham số.

Ghi chú các điểm paper chưa nêu rõ (đổi tại đây nếu cần):
  * F trong Alg3: paper không cho -> dùng F(x) = f(x) = 0.5 x  (bản gốc ACTAS dùng 7x).
  * Alg1: Table 1 ghi lam_k = a/ell và rho_k <= 2/(1 + a*theta_k), theta_k = lam_k/lam_{k+1}.
        stepsize_mode = "constant" (lam = a/ell, theta_k = 1)  hoặc "adaptive" (Bot et al., eq. 14).
        rho_mode      = "table" (rho_k = 2/(1 + a*theta_k))    hoặc "theory" (thỏa điều kiện (25) của Bot et al.).
  * Điểm khởi tạo Ex 4.1: xem problems.X0_CASES.
"""
import numpy as np

SEED = 42
TOL = 1e-5                     # err chung (Table 1)
MAX_ITER = {"4.1": 5_000, "4.2": 35_000, "4.3": 35_000, "4.4": 35_000}

# Hàm viscosity f(x) = 0.5 x  
f_visc = lambda x: 0.5 * x
# Toán tử F của Alg3 
F_alg3 = lambda x: 0.5 * x

# ---- các dãy tham số điều khiển chung (k >= 1) ------------------------------------------------
alpha_k = lambda k: 0.1 / (k + 1) ** 0.99
eta_k = lambda k: 0.1 / (k + 1) ** 1.1
tau_k = lambda k: k / (2 * k + 1)
mu_k = lambda k: k / (3 * k + 1)
inv_sq = lambda k: 1.0 / (k + 1) ** 2

PARAMS = {
    "ours": dict(lam1=0.5, theta=0.4, a=0.3, b=1.8,
                 alpha=alpha_k, eta=eta_k,
                 delta=lambda k: 1.0, xi=inv_sq, a_k=inv_sq, b_k=inv_sq),
    "alg1": dict(inertia=0.4, a=0.3, lam1=0.5,  
                 stepsize_mode="constant", rho_mode="table"),
    "alg2": dict(lam1=0.5, alpha=0.4, theta=0.4, mu=0.3),
    "alg3": dict(lam1=0.5, mu=0.3, alpha=alpha_k, eta=eta_k,
                 tau=tau_k, mu_k=mu_k, rho=inv_sq, xi=inv_sq),
    "alg4": dict(lam1=0.5, mu=0.3, alpha=alpha_k, eta=eta_k,
                 tau=tau_k, mu_k=mu_k),
}

LABELS = {
    "ours": "Our Alg.",
    "alg1": "Alg1 (Boţ et al.)",
    "alg2": "Alg2 (Cholamjiak et al.)",
    "alg3": "Alg3 (Thuy–Ly)",
    "alg4": "Alg4 (Thuy–Tung–Ly)",
}
METHODS = list(PARAMS)
