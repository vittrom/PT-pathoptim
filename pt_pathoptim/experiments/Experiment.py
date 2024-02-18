from helpers import createFolders
from initializers import *
from optimizers import *
from PT import DEO
from abc import abstractmethod

import numpy as np

class Experiment:
    """
    Represents an experiment.

    Args:
        args: The arguments for the experiment.
        deo_args: The arguments for the DEO PT algorithm.
        W: The log probability class.
    """

    def __init__(self, args, deo_args, W):
        self.deo_args = deo_args

        args_dict = vars(args)

        for key, value in args_dict.items():
            setattr(self, key, value)

        self.W = W(self.segments)

        self.foldername = self.W.__class__.__name__ + "_" + self.foldername
        self.foldername = "results/" + self.foldername
        createFolders(self.foldername)

        self.iters = self.optim_steps * self.K

        # Same computational budget but all on schedule, do 2*n every step
        if self.tune and not self.optim:
            self.optim_steps = int(np.log2(self.iters + 1) - 1)
            self.K = 2
        
        print("Optimize: " + str(self.optim))
        print("Tune: " + str(self.tune))
        print("Reversible " + str(self.reversible))

    @abstractmethod
    def init_state(self, n):
        """
        Initializes the state for the given value of n.

        Args:
            n: The value of n.

        Returns:
            The initialized state.
        """
        pass

    def run(self):
        """
        Runs the experiment.
        """
        for n in self.N:
            eta = init_eta(self.segments)
            beta = init_beta(n)
            for i in range(self.n_repl):
                current_state = self.init_state(n)

                pt = DEO(kernels=self.W.kernels(),
                         W=self.W.W_eta_segments(), W_KL=self.W.loss_KL_segments(),
                         gradient_fn=self.W.gradient_cov_segments(),
                         phi=eta, args=self.deo_args, optimizer=Adagrad(lr=self.lr), annealing_pars=beta)
                
                pt.simulate(iters=self.iters, init_state=current_state, burn_in=self.burn_in, K=self.K,
                            optim=self.optim, optim_iters=self.optim_steps,
                            savepath=self.foldername,
                            filename="results__n" + str(n) + "_repl_" + str(i),
                            tune=self.tune, reversible=self.reversible)

