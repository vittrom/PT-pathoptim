import autograd.numpy as np
from autograd import jacobian
from W.W import W

# state is a N x D vector of states for each chain
# etas is a N x 2 vector of natural parameters for the path; etas[:, 0] are the reference weights, etas[:, 1] are the target weights
def kernels(state, etas, args=None):
    mu_r = args[1]
    sigma_r = args[2]  # this is just a 1d numpy array SD NOT VARIANCE

    mu_t = args[3]
    sigma_t = args[4]  #same for this

    dims = state.shape[1]
    sds_r = np.repeat(sigma_r, dims)
    sds_t = np.repeat(sigma_t, dims)

    #We can broadcast operations assuming target and reference sigmas are the same diagonal matrix.
    sigma_etas = etas[:, 0][:, np.newaxis] * (1/sds_r**2)[np.newaxis, :] + etas[:, 1][:, np.newaxis] * (1/sds_t**2)[np.newaxis, :] # N chains x dims
    inv_sigma_etas = 1/sigma_etas # N_chains x dims
    mu_etas = etas[:, 0][:, np.newaxis] * (mu_r/(sds_r**2)) + etas[:, 1][:, np.newaxis] * (mu_t/(sds_t**2))  # N chains x dims
    mus = mu_etas * inv_sigma_etas
    state = np.random.normal(loc=mus, scale=np.sqrt(inv_sigma_etas)) # This should be N chains x dims, we can sample marginally since diag covariance

    return state

def W_endpoints(phi, beta, X=None, args=None):
    if beta == 0:
        mu = args[1]
        sigma = args[2]
    else:
        mu = args[3]
        sigma = args[4]
    dims = mu.shape[0]

    if X.ndim == 3:
        # first dimension is n_chains - 1
        # second dimension is dims
        # third dimension is n_samples
        # We want the result to be n_chains - 1 x n_samples
        # it is possible to do without loop but left for later

        # X: n_chains, dims, n_samples
        # mu: dims
        res = np.log(2 * np.pi) * dims / 2 + dims * np.log(sigma) + 0.5 * np.sum((X - mu[np.newaxis, :, np.newaxis])**2, 1)/sigma**2
        return res
    else:
        return + np.log(2 * np.pi) * dims/2 + dims * np.log(sigma) +\
           0.5 * np.sum((X - mu)**2, 1)/sigma**2

def W_eta(phi, beta, X=None, args=None, K=1):
    # K: #of segments in the path
    # phi = [eta_0_0, eta_0_1, eta_1_0, eta_1_1, ....]

    #check for when X is 3d

    W0 = W_endpoints(phi, 0, X, args)
    W1 = W_endpoints(phi, 1, X, args)
    if K == 1:
        W = (1 - beta) * W0 + beta * W1
    else:
        eta_K = phi
        etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta_K, np.array([0, 1]).reshape(-1, 2)), axis=0)
        if isinstance(beta, float):
            pass
        else:
            etas_annealing = np.zeros((len(beta), 2))

        for k in range(1, K + 1):
            if isinstance(beta, float):
                t = beta
                if t >= (k -1)/K and t <= k/K:
                    etas_annealing = etas[k - 1, :] * (k - K * t) + etas[k, :] * (t * K - k + 1)
                    W = W0 * etas_annealing[0] + W1 * etas_annealing[1]
                    return W
            else:
                if k == K:
                    etas_annealing += ((beta >= (k - 1) / K) & (beta <= k / K))[:, None] * (
                                (k - K * beta)[:, None] * etas[k - 1, :][None, :] +
                                (beta * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)
                else:
                    etas_annealing += ((beta >= (k - 1) / K) & (beta < k / K))[:, None] * ((k - K * beta)[:, None] * etas[k - 1, :][None, :] +
                                                                                 (beta * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)
        if len(W0.shape) > 1:
            W = W0 * etas_annealing[:, 0][:, None] + W1 * etas_annealing[:, 1][:, None]
        else:
            W = W0 * etas_annealing[:, 0] + W1 * etas_annealing[:, 1]
    return W

def W_KL_eta(phi, beta, X=None, args=None, K=1):
    X_0 = X[:, 0:-1, :]
    X_0 = np.moveaxis(X_0, 0, -1)
    X_1 = X[:, 1::, :]
    X_1 = np.moveaxis(X_1, 0, -1)

    sym_kl = np.mean(- W_eta(phi, beta[0:-1], X_0, args, K=K) + \
                     W_eta(phi, beta[1::], X_0, args, K=K) - \
                     W_eta(phi, beta[1::], X_1, args, K=K) + \
                     W_eta(phi, beta[0:-1], X_1, args, K=K), 1)
    return sym_kl

def loss_KL_eta(phi, beta, X=None, args=None,K=1):
    return np.sum(W_KL_eta(phi, beta, X, args, K=K))

def sqrts_KL_eta(phi, beta, X=None, args=None, K=1):
    return np.sum(np.sqrt(0.5 * W_KL_eta(phi, beta, X, args, K=K)))

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
    X = np.moveaxis(X, 0, -1)

    eta = phi
    n_pars = 2

    # set etas
    N = len(beta) #etas.shape[0]

    z_J = etas_zj(eta, beta, K)
    W0 = -W_endpoints(phi, 0, X, args).squeeze().T
    W1 = -W_endpoints(phi, 1, X, args).squeeze().T
    J = np.moveaxis(np.array([W0, W1]), 0, -1).reshape(-1, n_pars * N)
    W = J # dimension 2 x samples x Nchains
    E_J = np.mean(np.array([W0, W1]), 1).T.reshape(-1)
    g_zj = grad_zj(eta,beta, K).reshape(N * n_pars, -1)
    g_zw = grad_zw(eta.reshape(-1), beta, K)
    C = np.cov(J.T)
    grad = np.dot(np.dot(g_zw.T, C), z_J) + np.dot(g_zj.T, E_J)
    # print(grad)
    return grad

def W_eta_segments(K):
    def W_p(phi, beta, X=None, args=None):
        return W_eta(phi=phi, beta=beta, X=X, args=args, K=K)
    return W_p

def loss_KL_segments(K):
    def W_p(phi, beta, X=None, args=None):
        return loss_KL_eta(phi=phi, beta=beta, X=X, args=args, K=K)
    return W_p

def sqrts_KL_segments(K):
    def W_p(phi, beta, X=None, args=None):
        return sqrts_KL_eta(phi=phi, beta=beta, X=X, args=args,K=K)
    return W_p

def gradient_cov_segments(K):
    def W_p(phi, beta, X=None, args=None):
        return gradient_cov(phi, beta, X, args, W_endpoints=W_endpoints, K=K)
    return W_p

class W_mvn(W):
    def __init__(self, segments):
        self.segments = segments

    def kernels(self):
        return kernels

    def W_eta_segments(self):
        return W_eta_segments(self.segments)

    def loss_KL_segments(self):
        return loss_KL_segments(self.segments)

    def gradient_cov_segments(self):
        return gradient_cov_segments(self.segments)