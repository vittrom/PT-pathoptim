from gradient_fns import *
import copy

# state is a N x D vector of states for each chain
# etas is a N x 2 vector of natural parameters for the path; etas[:, 0] are the reference weights, etas[:, 1] are the target weights

def kernels(state, etas, args=None):
    mu_r = args[1]
    sd_r = args[2]

    mu_t = args[3]
    sd_t = args[4]

    sds = 1./(etas[:, 0]/sd_r**2 + etas[:, 1]/sd_t**2)
    mus = sds*(etas[:, 0]*mu_r/sd_r**2 + etas[:, 1]*mu_t/sd_t**2)
    sds = np.sqrt(sds)

    return (mus + np.random.randn(state.shape[0])*sds)[:,np.newaxis]

def W_endpoints(phi, beta, X=None, args=None):
    if beta == 0:
        mu = args[1]
        sd = args[2]
    else:
        mu = args[3]
        sd = args[4]
    return 0.5 * (X - mu) ** 2 / (sd ** 2) + np.log(sd) + 0.5 * np.log(2 * np.pi)

def W_eta(phi, beta, X=None, args=None, K=1):
    # K: #of segments in the path
    # phi = [eta_0_0, eta_0_1, eta_1_0, eta_1_1, ....]

    if X.ndim == 2 and X.shape[1] == 1:
        X = X.reshape(-1, )
    # if X.ndim == 2 and X.shape[1] > 1:
    #     beta = beta.reshape(-1, 1)

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
    X = X[:, :, 0]
    X_0 = X[:, 0:-1]
    X_0 = np.moveaxis(X_0, 0, -1)
    X_1 = X[:, 1::]
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
