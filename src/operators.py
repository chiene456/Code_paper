import numpy as np
from scipy.special import expit  


def soft_threshold(u, tau):
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)


def sigmoid(s):
    return expit(s)


def lambda_max_gram(T):
    return float(np.linalg.norm(T, 2) ** 2)
