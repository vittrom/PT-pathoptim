import autograd.numpy as np
import copy
import sys
from autograd import jacobian


# state is a N x D vector of states for each chain
# in the mixture model, the state is of the form [w1, w2, th1, th2, sig1, sig2, z1, z2, z3, z4... zN]
# etas is a N x 2 vector of natural parameters for the path; etas[:, 0] are the reference weights, etas[:, 1] are the target weights
def kernels(state, etas, args=None):
    modes = args[0]
    mus = np.array(args[1:(modes + 1)])[:, 0]
    sds = np.array(args[1:(modes + 1)])[:, 1]
    data = args[(2*modes + 1)]  # data to cluster

    ####
    # w step
    ####

    # gibbs step that depends on z -- mixes slowly
    # compute statistics
    # note that no data enter this term, so prior and posterior are identical

    inds = np.arange(modes, dtype=int)
    sumz = ((inds[:, np.newaxis, np.newaxis] == state[:, (3*modes):]).sum(axis=2)).T

    priora1 = np.repeat([np.repeat(1., modes)], state.shape[0], axis=0) - 1
    priora1 += sumz #Check correct
    posteriora1 = priora1
    # interpolate with etas
    aeta = etas[:, 0][:, np.newaxis] * priora1 + etas[:, 1][:, np.newaxis] * posteriora1 + 1
    # take a Gibbs w step
    state[:, :modes] = np.random.gamma(
        shape=aeta)  # can't use dirichlet with 2d array of params; use dirichlet = gamma + normalization
    state[:, :modes] /= state[:, :modes].sum(axis=1)[:, np.newaxis]

    ####
    # better MH w step that marginalizes z and avoids really slow random walk diffusion
    ####

    # compute the log likelihood for datapoints in each cluster
    loglikes = np.zeros((state.shape[0], state.shape[1] - (3 * modes), modes))
    for i in range(modes):
        loglikes[:, :, i] = -(state[:, (modes + i)][:, np.newaxis] - data) ** 2 / (2 * state[:, (2*modes + i)][:, np.newaxis] ** 2) - np.log(
            state[:, (2*modes + i)][:, np.newaxis])
    # get soft assignments to clusters, anneal, and use as dirichlet statistics
    llmax = loglikes.max(axis=2)
    likes = np.exp(loglikes - llmax[:, :, np.newaxis]) / (
        np.exp(loglikes - llmax[:, :, np.newaxis]).sum(axis=2)[:, :, np.newaxis])
    wstat = etas[:, 1][:, np.newaxis] * likes.sum(axis=1) + 1
    # generate an independent dirichlet proposal
    proposal = np.random.gamma(
        shape=wstat)  # can't use dirichlet with 2d array of params; use dirichlet = gamma + normalization
    proposal /= proposal.sum(axis=1)[:, np.newaxis]
    # compute the MH ratio
    # current log like
    cur_ll = np.log(state[:, :modes])[:, np.newaxis, :] + loglikes
    cllmax = cur_ll.max(axis=2)
    cur_ll = (np.log(np.exp(cur_ll - cllmax[:, :, np.newaxis]).sum(axis=2)) + cllmax).sum(axis=1)
    cur_ll_annealed = etas[:, 1] * cur_ll
    # proposal log like
    prop_ll = np.log(proposal)[:, np.newaxis, :] + loglikes
    pllmax = prop_ll.max(axis=2)
    prop_ll = (np.log(np.exp(prop_ll - pllmax[:, :, np.newaxis]).sum(axis=2)) + pllmax).sum(axis=1)
    prop_ll_annealed = etas[:, 1] * prop_ll
    # reverse transition + forward transition
    rev_ll = 0
    fwd_ll = 0
    for i in range(modes):
        rev_ll += (wstat[:, i] - 1) * np.log(state[:, i])
        fwd_ll += (wstat[:, i] - 1) * np.log(proposal[:, i])
    # accept/reject
    accepts = np.log(np.random.rand(state.shape[0])) <= prop_ll_annealed - cur_ll_annealed + rev_ll - fwd_ll
    state[accepts, :modes] = proposal[accepts, :modes]

    ####
    # label step
    ####
    # print('label move')
    # print('eta')
    # print(etas[20,:])
    # print('mu')
    # print(state[20, 2:4])
    # compute statistics
    # in the prior, points are just assigned based on weights
    logprior = np.zeros((state.shape[0], state.shape[1] - (3 * modes), modes))
    logprior[:, :, :] = np.log(state[:, :modes][:, np.newaxis, :])
    # in the posterior, the observation likelihood influences
    logpost = logprior + loglikes
    # interpolate with etas -- note that the prior is even probability
    logpetas = etas[:, 0][:, np.newaxis, np.newaxis] * logprior + etas[:, 1][:, np.newaxis, np.newaxis] * logpost
    logpetas -= logpetas.max(axis=2)[:, :, np.newaxis]
    logpetas[logpetas < -400] = -400  # -inf = -400 here to avoid underflow
    petas = np.exp(logpetas) / (np.exp(logpetas).sum(axis=2)[:, :, np.newaxis])
    # take the step
    state[:, (3*modes):] = (np.random.rand(*state[:, (3*modes):].shape)[:, :, np.newaxis] <= np.cumsum(petas, 2)[:, :, 0:2]).sum(axis=2)

    # print('sumz after ' + str(state[0, 6:].sum()))
    # print('mu before ' + str(state[0, 2:4]))

    ####
    # th step
    ####
    inds = np.arange(modes, dtype=int)
    sumz = ((inds[:, np.newaxis, np.newaxis] == state[:, (3 * modes):]).sum(axis=2)).T

    # get inverse variances statistics
    sds_prior = np.ones((state.shape[0], modes)) * 1/(sds ** 2)
    sds_post = np.zeros(sds_prior.shape)
    sds_post += sds_prior
    sds_post += sumz / state[:, (2*modes):(3*modes)] ** 2
    # interpolate
    sdetas = etas[:, 0][:, np.newaxis] * sds_prior + etas[:, 1][:, np.newaxis] * sds_post
    # invert to get variance (*NOTE* we sqrt sdetas later on to get std devs-- right now they're precisions)
    sdetas = 1. / sdetas
    # get mean statistics of ths (clever broadcasting hack since zs can only be = 0 or 1)
    mus_prior = np.ones((state.shape[0], modes)) * mus/(sds **2)
    mus_post = np.zeros(mus_prior.shape)
    mus_post += mus_prior
    # For now, there must be a better way
    for i in range(modes):
        mus_post[:, i] += (data * (state[:, (3*modes):] == i)).sum(axis=1) / state[:, (2*modes + i)] ** 2
    # interpolate
    muetas = etas[:, 0][:, np.newaxis] * mus_prior + etas[:, 1][:, np.newaxis] * mus_post
    # compute means and std devs of normals for ths
    muetas *= sdetas
    sdetas = np.sqrt(sdetas)
    # take the step
    state[:, modes:(2*modes)] = muetas + sdetas * np.random.randn(*sdetas.shape)

    # print('mu after ' + str(state[0, 2:4]))

    # state[:, 2] = 100
    # state[:, 3] = 200
    # print('means: ' + str(state[-1, 2:4]))

    return state


def W_normal(X, args=None):
    if args.shape[0] > 1:
        mu = args[:, 0]
        sigma = args[:, 1]
    else:
        mu = args[0]
        sigma = args[1]

    if X.ndim == 2:
        res = - 0.5 * (X - mu) ** 2 / (sigma * sigma) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        res = np.sum(res, 1)
    elif X.ndim == 3:
        res = - 0.5 * (X - mu[np.newaxis, :, np.newaxis]) ** 2 / (
                    sigma[np.newaxis, :, np.newaxis] * sigma[np.newaxis, :, np.newaxis]) - \
              np.log(sigma[np.newaxis, :, np.newaxis]) - 0.5 * np.log(2 * np.pi)

        res = np.sum(res, 1).squeeze()
    return - res


def W_categorical(X, args=None):
    modes = args[0]
    if X.ndim == 2:
        probs = X[:, :modes]

        inds = np.arange(modes, dtype=int)
        sumz = ((inds[:, np.newaxis, np.newaxis] == X[:, (3 * modes):]).sum(axis=2)).T

        res = (sumz * np.log(probs)).sum(axis=1)

    elif X.ndim == 3:
        probs = X[:, :modes, :]

        inds = np.arange(modes, dtype=int)
        sumz = ((inds[:, np.newaxis, np.newaxis, np.newaxis] == X[:, (3 * modes):, :]).sum(axis=2)).T
        sumz = np.moveaxis(sumz, 0, -1)
        res = (sumz * probs).sum(axis=1) #This should be N chains x n_obs
    return - res


def W_likelihood(X, args=None):
    modes = args[0]
    data = args[(2*modes+1)]
    if X.ndim == 2:
        means = X[:, modes:(2*modes)]
        sds = X[:, (2*modes):(3*modes)]
        indicators = X[:, (3*modes):]
        res = 0
        for i in range(modes):
            res += np.sum((indicators == i) * (
                    -0.5 * (data - means[:, i][:, np.newaxis]) ** 2 / (sds[:, i][:, np.newaxis] ** 2) - np.log(
                sds[:, i])[:, np.newaxis] - 0.5 * np.log(2 * np.pi)), 1)
    elif X.ndim == 3:
        means = X[:, modes:(2*modes), :]
        sds = X[:, (2*modes):(3*modes), :]
        indicators = X[:, (3*modes):, :]
        res = 0
        for i in range(modes):
            res += np.sum((indicators == i) * (-0.5 * (data[np.newaxis, :, np.newaxis] - means[:, i, :][:, np.newaxis, :]) ** 2 / (
                    sds[:, i, :][:, np.newaxis, :] ** 2) - np.log(sds[:, i, :][:, np.newaxis, :]) - 0.5 * np.log(
            2 * np.pi)), 1) #Check that this is indeed correct

    return -res


def W_endpoints(phi, beta, X=None, args=None):
    modes = args[0]
    if X.ndim == 2:
        means = X[:, modes:(2 * modes)]
    elif X.ndim == 3:
        means = X[:, modes:(2 * modes), :]
    if beta == 0:
        V0 = W_normal(means, np.array(args[1:(modes+1)])) + \
             W_categorical(X, args)
        return V0
    else:
        V0 = W_normal(means, np.array(args[1:(modes+1)])) + \
             W_categorical(X, args)
        V1 = V0 + W_likelihood(X, args)
        return V1


def W_eta(phi, beta, X=None, args=None, K=1):
    # K: #of segments in the path
    # phi = [eta_0_0, eta_0_1, eta_1_0, eta_1_1, ....]

    W0 = W_endpoints(phi, 0, X, args)
    W1 = W_endpoints(phi, 1, X, args)
    if K == 1:
        W = (1 - beta) * W0 + beta * W1
    else:
        eta_K = phi
        etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta_K, np.array([0, 1]).reshape(-1, 2)), axis=0)
        etas_annealing = np.zeros((len(beta), 2))

        for k in range(1, K + 1):
            if k == K:
                etas_annealing += ((beta >= (k - 1) / K) & (beta <= k / K))[:, None] * (
                        (k - K * beta)[:, None] * etas[k - 1, :][None, :] +
                        (beta * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)
            else:
                etas_annealing += ((beta >= (k - 1) / K) & (beta < k / K))[:, None] * (
                            (k - K * beta)[:, None] * etas[k - 1, :][None, :] +
                            (beta * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)
        if X.ndim == 3:
            W = W0 * etas_annealing[:, 0][:, np.newaxis] + W1 * etas_annealing[:, 1][:, np.newaxis]
        else:
            W = W0 * etas_annealing[:, 0] + W1 * etas_annealing[:, 1]
    return W


def W_KL_eta(phi, beta, X=None, args=None, K=1):
    X_0 = X[:, 0:-1, :]
    X_0 = np.moveaxis(X_0, 0, -1)
    X_1 = X[:, 1::, :]
    X_1 = np.moveaxis(X_1, 0, -1)

    sym_kl = np.mean(-W_eta(phi, beta[0:-1], X_0, args, K=K) + \
                     W_eta(phi, beta[1::], X_0, args, K=K) - \
                     W_eta(phi, beta[1::], X_1, args, K=K) + \
                     W_eta(phi, beta[0:-1], X_1, args, K=K), 1)
    return sym_kl


def loss_KL_eta(phi, beta, X=None, args=None, K=1):
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

    etas_annealing += (betas == 1)[:, None] * np.array([0., 1.])

    z_j = np.zeros((len(betas), 2))
    for i in range(N):
        if i == 0:
            z_j += (np.arange(N) == i)[:, None] * (etas_annealing[0, :] - etas_annealing[1, :])
        elif i == (N - 1):
            z_j += (np.arange(N) == i)[:, None] * (etas_annealing[-1, :] - etas_annealing[-2, :])
        else:
            z_j += (np.arange(N) == i)[:, None] * (
                        2 * etas_annealing[i, :] - etas_annealing[i - 1, :] - etas_annealing[i + 1, :])

    return z_j.reshape(-1, )


def etas_w(eta, betas, K=2):
    etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta.reshape(-1, 2), np.array([0, 1]).reshape(-1, 2)),
                          axis=0)
    etas_annealing = np.zeros((len(betas), 2))
    for k in range(1, K + 1):
        etas_annealing += ((betas >= (k - 1) / K) & (betas < k / K))[:, None] * (
                (k - K * betas)[:, None] * etas[k - 1, :][None, :] +
                (betas * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)

    etas_annealing += (betas == 1)[:, None] * np.array([0., 1.])
    return etas_annealing.reshape(-1, )


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
    N = len(beta)  # etas.shape[0]

    z_J = etas_zj(eta, beta, K)
    W0 = -W_endpoints(phi, 0, X, args).squeeze().T
    W1 = -W_endpoints(phi, 1, X, args).squeeze().T
    J = np.moveaxis(np.array([W0, W1]), 0, -1).reshape(-1, n_pars * N)
    W = J  # dimension 2 x samples x Nchains
    E_J = np.mean(np.array([W0, W1]), 1).T.reshape(-1)
    g_zj = grad_zj(eta, beta, K).reshape(N * n_pars, -1)
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
        return sqrts_KL_eta(phi=phi, beta=beta, X=X, args=args, K=K)

    return W_p


def gradient_cov_segments(K):
    def W_p(phi, beta, X=None, args=None):
        return gradient_cov(phi, beta, X, args, W_endpoints=W_endpoints, K=K)

    return W_p