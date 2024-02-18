import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), "pt_pathoptim"))

import argparse
from W.W_mvn import W_mvn
from helpers import str2bool
from Experiment import Experiment
import numpy as np


parser = argparse.ArgumentParser()
# Basic inputs
parser.add_argument('--reversible', type=str2bool, default=False,
help="Set to true for reversible PT")
parser.add_argument('--N', nargs="+", type=int, default=[240],
help="Number of chains")
parser.add_argument('--dims', type=int, default=20,
help="Dimensions of mvn")
parser.add_argument('--K', type=int, default=100, # [100, 1000, 5000, 10000, 25000, 50000, 100000],
help="Number of samples to use for gradient approximation")
parser.add_argument('--n_repl', type=int, default=1,
help="Number of replications")
parser.add_argument('--burn_in', type=int, default=0,
                    help="Burn in")
parser.add_argument('--tune', type=str2bool, default=True,
help="Schedule tuning rounds")
parser.add_argument('--optim_steps', type=int, default=500, #Probably will need more iterations
help="Number of optimization steps")
parser.add_argument('--optim', type=str2bool, default=True, #Set to False for NRPT
help="Do gradient optimization")
parser.add_argument('--foldername', type=str, default=os.path.join("Results", "test_mvn", "4_segments_100d"),
help="Name folder to save results")
parser.add_argument('--lr', type=float, default=0.2, #This seems to work fine
help="Learning rate")
parser.add_argument('--checkpoint_iter', type=int, default=100,
help="Number of iter before checkpoint")
parser.add_argument('--mu0', type=float, default=-1,
help="Mean pi0")
parser.add_argument('--mu1', type=float, default=1,
help="Mean pi1")
parser.add_argument('--sigma0', type=float, default=0.1,
help="Sigma pi0")
parser.add_argument('--sigma1', type=float, default=0.1,
help="Sigma pi1")
parser.add_argument('--segments', type=int, default=4,
help="Number of knots in spline")
arg = parser.parse_args()


dims = arg.dims
deo_args = [None, np.repeat(arg.mu0, dims), arg.sigma0, np.repeat(arg.mu1, dims), arg.sigma1]

class Experiment_mvn(Experiment):
    def __init__(self, args, deo_args, W):
        super().__init__(args, deo_args, W)

    def init_state(self, n):
        return np.array([np.random.normal(size=dims) for _ in range(n)])

experiment = Experiment_mvn(arg, deo_args, W_mvn)
experiment.run()