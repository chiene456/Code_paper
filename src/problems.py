\
import numpy as np

from .operators import soft_threshold, sigmoid, lambda_max_gram


def _euclid():
    return dict(inner=lambda u, v: float(np.dot(u, v)),
                norm=lambda u: float(np.linalg.norm(u)))


# --------------------------------------------------------------------------
# Example 4.1: SFP
# --------------------------------------------------------------------------
# Điểm khởi tạo:
X0_CASES = {
    "t2_10": lambda t: t ** 2 / 10,
    "2t_16": lambda t: 2.0 ** t / 16,                      
    "trig": lambda t: 2 * np.sin(5 * t) ** 4 - 3 * np.cos(-2 * t),
}

def sfp_l2_problem(N=1000, x0_case="t2_10"):
    t = np.linspace(0.0, 2 * np.pi, N)
    h = t[1] - t[0]
    w = np.full(N, h)
    w[0] = w[-1] = h / 2                                   
    inner = lambda u, v: float(np.sum(w * u * v))
    norm = lambda u: float(np.sqrt(inner(u, u)))

    one, sin_t = np.ones(N), np.sin(t)
    len_one = inner(one, one)                              

    def proj_C(z):                                          
        s = inner(one, z)
        return z + (1.0 - s) / len_one * one if s > 1 else z

    def proj_Q(z):                                         
        d = z - sin_t
        nd = norm(d)
        return sin_t + 4.0 / nd * d if nd > 4 else z

    return dict(name="Ex4.1-SFP-L2", B=lambda z: z - proj_Q(z),
                J=lambda z, lam: proj_C(z), ell=1.0,
                x0=X0_CASES[x0_case](t), inner=inner, norm=norm, t=t)


# --------------------------------------------------------------------------
# Example 4.2: LASSO
# --------------------------------------------------------------------------
def lasso_problem(K=512, L=1024, nnz=30, sigma_noise=0.01, sigma1=0.1, seed=42):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((K, L))
    u_true = np.zeros(L)
    idx = rng.choice(L, nnz, replace=False)
    u_true[idx] = rng.standard_normal(nnz)
    y = T @ u_true + sigma_noise * rng.standard_normal(K)
    x0 = rng.standard_normal(L)
    return dict(name="Ex4.2-LASSO",
                B=lambda x: T.T @ (T @ x - y),
                J=lambda z, lam: soft_threshold(z, lam * sigma1),
                ell=lambda_max_gram(T), x0=x0, u_true=u_true, **_euclid())


# --------------------------------------------------------------------------
# Example 4.3: Elastic Net
# --------------------------------------------------------------------------
def elastic_net_problem(K=512, L=1024, sigma1=0.1, sigma2=0.2, seed=42):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((K, L))
    z = rng.standard_normal(K)
    x0 = rng.standard_normal(L)
    return dict(name="Ex4.3-ElasticNet",
                B=lambda x: T.T @ (T @ x - z) + 2 * sigma2 * x,
                J=lambda v, lam: soft_threshold(v, lam * sigma1),
                ell=lambda_max_gram(T) + 2 * sigma2, x0=x0, **_euclid())


# --------------------------------------------------------------------------
# Example 4.4: Sparse logistic regression  
# --------------------------------------------------------------------------
def sparse_logistic_problem(L=512, K=1024, nnz=30, sigma1=0.1, seed=42):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((L, K))
    beta_true = np.zeros(K)
    idx = rng.choice(K, nnz, replace=False)
    beta_true[idx] = rng.standard_normal(nnz)
    y = rng.binomial(1, sigmoid(T @ beta_true)).astype(float)
    x0 = rng.standard_normal(K)
    return dict(name="Ex4.4-SparseLogReg",
                B=lambda b: T.T @ (sigmoid(T @ b) - y),
                J=lambda v, lam: soft_threshold(v, lam * sigma1),
                ell=lambda_max_gram(T) / 4.0, x0=x0, beta_true=beta_true, **_euclid())


BUILDERS = {
    "4.1": sfp_l2_problem,
    "4.2": lasso_problem,
    "4.3": elastic_net_problem,
    "4.4": sparse_logistic_problem,
}
