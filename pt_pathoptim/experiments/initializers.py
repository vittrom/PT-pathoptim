import numpy as np


def init_eta(segments):
    eta = np.concatenate(((1 - np.arange(1, segments) / segments).reshape(-1, 1),
                              (np.arange(1, segments) / segments).reshape(-1, 1)), axis=1)
    return np.log(eta)

def init_beta(n):
    beta = np.arange(n) / (n - 1)
    return beta