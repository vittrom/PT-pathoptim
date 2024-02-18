import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), "pt_pathoptim"))

import argparse
from W.W_betabinomial import W_betabinomial
from Experiment import Experiment
from helpers import str2bool
import numpy as np

parser = argparse.ArgumentParser()
# Basic inputs
parser.add_argument('--reversible', type=str2bool, default=False,
                    help="Set to true for reversible PT")
parser.add_argument('--N', nargs="+", type=int, default=[50],
                    help="Number of chains")
parser.add_argument('--K', type=int, default=100,
                    help="Number of samples to use for gradient approximation")
parser.add_argument('--n_repl', type=int, default=10,
                    help="Number of replications")
parser.add_argument('--burn_in', type=int, default=0,
                    help="Burn in")
parser.add_argument('--tune', type=str2bool, default=True,
                    help="Schedule tuning rounds")
parser.add_argument('--optim_steps', type=int, default=80,
                    help="Number of optimization steps")
parser.add_argument('--optim', type=str2bool, default=True,
                    help="Do gradient optimization")
parser.add_argument('--foldername', type=str, default=os.path.join("Results", "test_beta_binom", "nrpt"),
                    help="Name folder to save results")
parser.add_argument('--lr', type=float, default=1.,
                    help="Learning rate")
parser.add_argument('--checkpoint_iter', type=int, default=100,
                    help="Number of iter before checkpoint")
parser.add_argument('--alpha_reference', type=float, default=180,
                    help="Alpha")
parser.add_argument('--beta_reference', type=float, default=840,
                    help="Mean pi1")
parser.add_argument('--p', type=float, default=0.7,
                    help="Binomial prob")
parser.add_argument('--samps', type=int, default=2000,
                    help="Samples from binomial")
parser.add_argument('--n_binom', type=int, default=100,
                    help="N binomial")
parser.add_argument('--segments', type=int, default=2, # For 1 segment use 2 segments with optim False
                    help="Number of knots in spline")
parser.add_argument('--seed', type=int, default=12345,
                    help="Seed")
arg = parser.parse_args()

np.random.seed(arg.seed)
data = 70*np.ones(2000) #np.random.binomial(arg.n_binom, arg.p, size=arg.samps)
deo_args= [
    [arg.alpha_reference, arg.beta_reference],
    arg.n_binom,
    data
]

class Experiment_betabinomial(Experiment):
    def __init__(self, args, deo_args, W):
        super().__init__(args, deo_args, W)
    def init_state(self, n):
        return np.array([np.array([0.5]).reshape(-1,) for _ in range(n)])

experiment = Experiment_betabinomial(arg, deo_args, W_betabinomial)
experiment.run()