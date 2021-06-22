import os
from optimizers import Adagrad
from utils import *
import autograd.numpy as np
from scipy.interpolate import PchipInterpolator
import copy

class DEO():

    def __init__(self, kernels, W, gradient_fn, W_KL, phi, args,
                 annealing_pars,num_extra_params=0, n_expl=1, optimizer=Adagrad(lr=0.05)):
        self.num_extra_params = num_extra_params
        self.iter_first_rt = 0
        self.stats = []
        self.phi = phi
        if self.num_extra_params == 0:
            self.phi_linear = np.exp(phi)
        else:
            self.phi_linear = np.exp(phi[:-self.num_extra_params].reshape(-1, 2))
        self.num_replica = annealing_pars.shape[0]
        self.W = W
        self.W_KL = W_KL
        self.gradient_fn = gradient_fn
        self.current_iter = 0
        self.args = args + [self.num_replica]
        self.betas = annealing_pars

        self.kernels = kernels
        self.n_expl = n_expl
        self.etas = self.compute_etas()

        self.extra_pars = None
        self.update = False

        #set optimizer and options
        self.optimizer = optimizer

        #variables to store results (temporary)
        self.rej_stat = np.zeros(shape=self.num_replica - 1)
        self.acc_trunc = np.zeros(shape=self.num_replica - 1)
        self.rejs = []
        self.total_round_trips = []
        self.rejection_rates = []
        self.round_trip_stats = []

        # Round trips statistics
        self.position_indicators = np.arange(self.num_replica)
        self.prior_start = np.zeros(self.num_replica)
        self.prior_start[0] = 1
        self.posterior_end = np.zeros(self.num_replica)
        self.round_trips = 0

    def update_params(self, gradient):
        self.phi = self.optimizer.update(beta=self.phi, gradient=gradient)
        # Check monotonicity
        if self.num_extra_params == 0:
            self.phi_linear = np.exp(self.phi)
        else:
            self.phi_linear = np.exp(self.phi[:-self.num_extra_params]).reshape(-1, 2)
            self.args[0:self.num_extra_params] = self.phi[-self.num_extra_params:].tolist()

        self.phi_linear[:, 0] = fix_monotonicity_2(np.hstack((1., self.phi_linear[:, 0], 0.)))[1:-1]
        self.phi_linear[:, 1] = fix_monotonicity_2(np.hstack((0., self.phi_linear[:, 1], 1.)))[1:-1]

        if self.num_extra_params == 0:
            self.phi = np.log(self.phi_linear)
        else:
            self.phi[:-self.num_extra_params] = np.log(self.phi_linear).reshape(-1,)

    def initialize_round_trips_stats(self):
        self.position_indicators = np.arange(self.num_replica)
        self.prior_start = np.zeros(self.num_replica)
        self.prior_start[0] = 1
        self.posterior_end = np.zeros(self.num_replica)
        self.round_trips = 0.

    def update_round_trips_counter(self, A, S):
        # chains for which is true have a swap
        cond = A[S] == True
        # Update indicators
        tmp = self.position_indicators[S[cond]]
        self.position_indicators[S[cond]] = self.position_indicators[S[cond] + 1]
        self.position_indicators[S[cond] + 1] = tmp

        # Updates if a new sample starts from the prior or a sample from the prior got to the posterior
        # Check if any round trip has been concluded
        self.prior_start[self.position_indicators[0]] = 1
        self.posterior_end[self.position_indicators[-1]] = 1 if self.prior_start[self.position_indicators[-1]] == 1\
            else self.posterior_end[self.position_indicators[-1]]
        # Finished round trip
        self.round_trips += (self.prior_start[self.position_indicators[0]] == 1) & \
                            (self.posterior_end[self.position_indicators[0]] == 1)
        self.posterior_end[self.position_indicators[0]] = 0 if self.prior_start[self.position_indicators[0]] == 1 \
            else self.posterior_end[self.position_indicators[0]]

    def update_schedule(self):
        rejections = np.mean(np.array(self.rejs), 0)
        cum_rejections = np.concatenate((np.array([0]), np.cumsum(rejections)))
        gcb = cum_rejections[-1]
        cumulativebarrier = PchipInterpolator(x=self.betas, y=cum_rejections)
        betas = np.zeros(self.num_replica)
        betas[-1] = 1
        for i in range(1, self.num_replica - 1):
            f = lambda x: cumulativebarrier(x) - (gcb * (i/(self.num_replica - 1)))
            betas[i] = bisection(f, np.max((0, betas[i - 1] - 0.1)), 1)
        self.betas = betas
        self.etas = self.compute_etas()

    def compute_etas(self):
        if len(self.phi_linear) == 0:
            return self.betas
        else:
            K = len(self.phi_linear) + 1
            eta_K = self.phi_linear
            etas = np.concatenate((np.array([1, 0]).reshape(-1, 2), eta_K, np.array([0, 1]).reshape(-1, 2)), axis=0)
            etas_annealing = np.zeros((len(self.betas), 2))
            for k in range(1, K + 1):
                etas_annealing += ((self.betas >= (k - 1) / K) & (self.betas < k / K))[:, None] * (
                        (k - K * self.betas)[:, None] * etas[k - 1, :][None, :] +
                        (self.betas * K - k + 1)[:, None] * etas[k, :][None, :]).reshape(-1, 2)

            etas_annealing[-1, :] = np.array([0., 1.])

        if self.num_extra_params == 0:
            return etas_annealing
        else:
            return np.concatenate((etas_annealing.reshape(-1,), self.phi[-self.num_extra_params:]))

    def exploration_step(self, current_state):
        for k in range(self.n_expl): # do n_expl exploration steps
            current_state = self.kernels(current_state, self.etas, self.args)
            self.update = False
        return current_state

    def communication_step(self, current_state, reversible=False):
        indices_trunc = np.zeros(self.num_replica - 1)
        is_swapped = np.zeros(self.num_replica - 1)

        if not reversible:
            if self.current_iter % 2 == 0:
                S = np.arange(self.num_replica - 1)[0::2]
            else:
                S = np.arange(self.num_replica - 1)[1::2]
        else:
            if np.random.uniform(size=1) <= 0.5:
                S = np.arange(self.num_replica - 1)[0::2]
            else:
                S = np.arange(self.num_replica - 1)[1::2]
        # Compute acceptance probability
        if self.num_extra_params == 0:
            logalpha = compute_log_ratio(self.W, current_state,
                              self.phi_linear, self.betas, self.args).reshape(-1,)  # Useless to reshape but for now keep as is
        else:
            logalpha = compute_log_ratio(self.W, current_state,
                                         np.concatenate((self.phi_linear.reshape(-1,), self.phi[-self.num_extra_params:])),
                                         self.betas, self.args).reshape(-1,)  # Useless to reshape but for now keep as is

        # keep track of rejections statistics
        logalpha[np.isnan(logalpha) | (logalpha < -400)] = -400 # use -400 as -inf here since exp(-400) = 1e-174
        self.acc_trunc[logalpha > 0] += 1
        indices_trunc[logalpha > 0] = 1
        logalpha[logalpha > 0] = 0
        self.rej_stat += 1 - np.exp(logalpha)
        self.rejs.append(1 - np.exp(logalpha))

        # Swap states for which log(U) < alpha
        A = np.random.uniform(low=0, high=1, size=logalpha.shape[0]) < np.exp(logalpha)
        cond = A[S] == True
        is_swapped[S[cond]] = 1
        tmp = current_state[S[cond]]
        current_state[S[cond]] = current_state[S[cond] + 1]
        current_state[S[cond] + 1] = tmp

        return current_state, indices_trunc, is_swapped, A, S

    def step_pt(self, current_state, reversible=False):
        # print(current_state)
        # Exploration step
        current_state = self.exploration_step(current_state)

        # Swap step
        current_state, indices_trunc, is_swapped, A, S = self.communication_step(current_state,
                                                                                 reversible=reversible)

        self.update_round_trips_counter(A, S)
        self.current_iter += 1

        return current_state, indices_trunc, is_swapped

    def step_optim_phi_KL(self, grad):
        self.update_params(gradient=grad)
        self.etas = self.compute_etas()

    def simulate(self, iters, init_state, burn_in=1000, K=100, checkpoint_iter=1000,
                 KL=False, optim=True, optim_iters=1000,
                 savepath=None, filename=None, tune=True, reversible=False):

        # In here need to handle storing of states and truncation matrix and update of kernel results
        # Init state is a list of tensors, this is OK but be careful, you need to store list of arrays
        # betas_arr = [self.betas]
        etas_arr = [self.etas]
        result_phi = []
        kl_states = []
        k = 0
        current_state = init_state
        optim_iter = 0
        self.initialize_round_trips_stats()

        for i in range(iters):
            self.total_round_trips.append(self.round_trips)
            if self.iter_first_rt == 0 and self.round_trips > 0:
                self.iter_first_rt = i
            if i < burn_in:
               current_state, _, _ = self.step_pt(current_state=current_state,
                                                  reversible=reversible)
            else:
                k += 1
                current_state, indices_trunc, is_swapped = self.step_pt(current_state=current_state,
                                                                        reversible=reversible)

                temp = swap_back(current_state, is_swapped, self.num_replica)
                kl_states.append(np.copy(temp))

                # Update covariance if not KL optimization
                if i == (iters - 1) and tune and not optim:
                    self.update_stats(kl_states, k, KL=optim, iters=i)
                    self.do_plots(optim_iter, savepath)
                    self.rejs = []
                    self.rej_stat[:] = 0
                    print("Phi")
                    print(self.phi_linear)

                if k == K:
                    print("-------------------------------------------------")
                    print("Iteration: " + str(optim_iter))
                    self.update_stats(kl_states, k, KL=optim, iters=i)
                    print("Betas")
                    # print(self.betas)
                    self.do_plots(optim_iter, savepath)
                    # Do gradient step with current betas
                    if optim:
                        if self.num_extra_params == 0:
                            grad = self.gradient_fn(np.exp(self.phi), self.betas, np.array(kl_states[-k::]), self.args).reshape(-1, 2)
                            grad /= (np.fabs(grad) + np.exp(self.phi)) # newton method with H approx I
                        else:
                             tmp = copy.deepcopy(self.phi)
                             tmp[:-self.num_extra_params] = np.exp(tmp[:-self.num_extra_params])
                             grad = self.gradient_fn(tmp, self.betas, np.array(kl_states[-k::]),
                                                    self.args)
                             grad[:-self.num_extra_params] /= (np.fabs(grad[:-self.num_extra_params]) + np.exp(tmp[:-self.num_extra_params]))


                        print("Gradient")
                        self.step_optim_phi_KL(grad)

                    # Do schedule tuning/update betas
                    if tune:
                        self.update_schedule()
                        # betas_arr.append(np.copy(self.betas))
                        etas_arr.append(self.etas)
                        print("Samples used for estimate:" + str(K))
                        if not optim:
                            K = 2 * K  # Double samples
                    else: #if no beta update required, the path changed, so we still need to update eta
                        etas_arr.append(self.etas)

                    optim_iter += 1
                    self.rejs = []
                    self.rej_stat[:] = 0
                    print("Phi")
                    print(self.phi)
                    if self.num_extra_params == 0:
                        print(np.exp(self.phi))
                    else:
                        print(np.exp(self.phi[:-self.num_extra_params]))

                    k = 0
                    kl_states = []
                    if optim and optim_iter == optim_iters:
                        optim = False


            if (i % checkpoint_iter == 0 and i > 0) or i == (iters - 1):
                np.savez(os.path.join(savepath, filename),
                         pars=result_phi,
                         round_trip_stats=np.array(self.round_trip_stats),
                         cumulative_round_trips=np.array(self.total_round_trips),
                         args=self.args,
                         etas=etas_arr)

    def update_stats(self, kl_states, k, KL=True, iters=1):
        rejections = np.mean(np.array(self.rejs), 0)
        skl_val = 0
        if not KL:
            pass
        else:
            if self.num_extra_params == 0:
                skl_val = self.W_KL(self.phi_linear, self.betas,
                          np.array(kl_states[-k::]),
                          self.args)
            else:
                skl_val = self.W_KL(np.concatenate((self.phi_linear.reshape(-1,), self.phi[-self.num_extra_params:])), self.betas,
                          np.array(kl_states[-k::]),
                          self.args) if KL else 0
        self.round_trip_stats.append([np.sum(rejections),   #GCB
                                      np.sum(rejections / (1 - rejections)) if np.all(rejections < 1) else np.inf,  #Efficiency
                                      np.std(rejections), #STD rejections
                                      1 / (2 + 2 * np.sum(rejections/(1-rejections))) if np.all(rejections < 1) else 0., #NARTS
                                      self.round_trips, #Round trips
                                      skl_val])

        print("AsympGCB: " + str(self.round_trip_stats[-1][0]))
        print("NonAsympGCB: " + str(self.round_trip_stats[-1][1]))
        print("SKL: " + str(self.round_trip_stats[-1][5]))
        print("Mean rejections: " + str(np.mean(rejections)))
        print("SD rejections: " + str(self.round_trip_stats[-1][2]))
        print("Non-asymptotic round trips:  " + str(self.round_trip_stats[-1][3]))
        print("Round trips: " + str(self.round_trips))
        print("Round trip rate:" + str(self.round_trips/(iters + 1 - self.iter_first_rt)))

    def do_plots(self, optim_iter, savepath):
        rejections = np.mean(np.array(self.rejs), 0)
        args_plot = [optim_iter, np.mean(rejections), self.round_trip_stats[-1][5],
                     self.round_trip_stats[-1][3], self.round_trip_stats[-1][4],
                     rejections]
        args_names = ["Iter", "Mean rejs", "SKL", "NARTR", "RTs"]

        plot_etas(self.etas, self.phi_linear, savepath=os.path.join(savepath, "path", f"plot_etas_{optim_iter:04d}.png"),
                  args=args_plot, args_names=args_names, extra_pars=self.num_extra_params)
        plot_etas_log(self.etas, self.phi_linear, savepath=os.path.join(savepath, "path_log", f"plot_etas_{optim_iter:04d}.png"),
                  args=args_plot, args_names=args_names, extra_pars=self.num_extra_params)
