import autograd.numpy as np
from autograd import jacobian

# Gradient functions for the path optimization problem with normal targets, see paper for more info.

def etas_zj(eta, betas, K=2):
        N = len(betas)

        etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta, np.array([0, 1]).reshape(-1, 2)), axis=0)

        etas_annealing = np.zeros((len(betas), 2))

        for k in range(1, K + 1):
            etas_annealing += ((betas >= (k - 1) / K) & (betas < k / K))[:, None] * (
                    (k - K * betas)[:, None] * etas[k - 1, :][None, :] +
                    (betas * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)

        etas_annealing += (betas ==  1)[:, None] * np.array([0., 1.])


        z_j = np.zeros((len(betas), 2))
        for i in range(N):
            if i == 0:
                z_j += (np.arange(N) == i)[:, None] * (etas_annealing[0, :] - etas_annealing[1, :])
            elif i == (N -1):
                z_j += (np.arange(N) == i)[:, None] * (etas_annealing[-1, :] - etas_annealing[-2, :])
            else:
                z_j += (np.arange(N) == i)[:, None] * (2 * etas_annealing[i, :] - etas_annealing[i - 1, :] - etas_annealing[i + 1, :])

        return z_j.reshape(-1,)

def etas_w(eta,  betas, K=2):
        etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta.reshape(-1, 2), np.array([0, 1]).reshape(-1, 2)), axis=0)
        etas_annealing = np.zeros((len(betas), 2))
        for k in range(1, K + 1):
            etas_annealing += ((betas >= (k - 1) / K) & (betas < k / K))[:, None] * (
                    (k - K * betas)[:, None] * etas[k - 1, :][None, :] +
                    (betas * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)

        etas_annealing += (betas ==  1)[:, None] * np.array([0., 1.])
        return etas_annealing.reshape(-1,)

def grad_zw(eta, betas, K=2):
        fn = jacobian(etas_w)
        return fn(eta, betas, K)

def grad_zj(eta, betas, K=2):
    fn = jacobian(etas_zj)
    return fn(eta, betas, K)

def gradient_cov(phi, beta, X, args, W_endpoints, K=2):
        eta = phi
        n_pars = 2

        # set etas
        etas = np.concatenate((np.array([1, 0]).reshape(-1, n_pars), eta, np.array([0, 1]).reshape(-1, n_pars)))
        N = len(beta) #etas.shape[0]

        z_J = etas_zj(eta, beta, K)
        W0 = -W_endpoints(phi, 0, X, args).squeeze()
        W1 = -W_endpoints(phi, 1, X, args).squeeze()
        J = np.moveaxis(np.array([W0, W1]), 0, -1).reshape(-1, n_pars * N)
        W = J # dimension 2 x samples x Nchains
        E_J = np.mean(np.array([W0, W1]), 1).T.reshape(-1)
        g_zj = grad_zj(eta, beta, K).reshape(N * n_pars, -1)
        g_zw = grad_zw(eta.reshape(-1), beta, K)
        C = np.cov(J.T)
        grad = np.dot(np.dot(g_zw.T, C), z_J) + np.dot(g_zj.T, E_J)
        # print(grad)
        return grad
