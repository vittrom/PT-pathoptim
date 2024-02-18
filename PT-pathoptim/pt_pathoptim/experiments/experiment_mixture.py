import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), "pt_pathoptim"))

import argparse
from W.W_mixture import W_mixture
import numpy as np
from Experiment import Experiment
from helpers import str2bool

parser = argparse.ArgumentParser()
# Basic inputs
parser.add_argument('--reversible', type=str2bool, default=False,
                    help="Set to true for reversible PT")
parser.add_argument('--N', nargs="+", type=int, default=[10],
                    help="Number of chains")
parser.add_argument('--K', type=int, default=100,
                    help="Number of samples to use for gradient approximation")
parser.add_argument('--n_repl', type=int, default=10,
                    help="Number of replications")
parser.add_argument('--burn_in', type=int, default=100,
                    help="Burn in")
parser.add_argument('--tune', type=str2bool, default=True,
                    help="Schedule tuning rounds")
parser.add_argument('--optim_steps', type=int, default=500,
                    help="Number of optimization steps")
parser.add_argument('--optim', type=str2bool, default=False,
                    help="Do gradient optimization")
parser.add_argument('--foldername', type=str, default=os.path.join("Results", "test_multimodal_mixture", "2_segments"),
                    help="Name folder to save results")
parser.add_argument('--lr', type=float, default=.3,
                    help="Learning rate")
parser.add_argument('--checkpoint_iter', type=int, default=100,
                    help="Number of iter before checkpoint")
parser.add_argument('--datapath', type=str, default=os.path.join("pt_pathoptim", "data", "galaxies.csv"),
                    help="Path to data")
parser.add_argument('--modes', type=int, default=6, # For 1 segment use 2 segments with optim False
                    help="Number of modes")
parser.add_argument('--segments', type=int, default=3, # For 1 segment use 2 segments with optim False
                    help="Number of knots in spline")
arg = parser.parse_args()

# Load data
data = np.genfromtxt(arg.datapath, skip_header=-1, delimiter=",")
data_len = data.shape[0]
modes = arg.modes

#true modes = data generation
# 50 points of N(100, 5^2)
# 150 points of N(200, 10^2)
# so w approx (.25, .75)
# mu (100, 200)
# sd (5, 10)

# args = [[mu0, sd0], [mu1, sd1], [a0, b0], [a1, b1], data] (a0/1, b0/1 are shape/scale for inverse gamma)
deo_args = [modes] + modes * [[150., 1.]] + modes * [[1.1, 10]] + [data]


class Experiment_mixture(Experiment):
    def __init__(self, args, deo_args, W):
        super().__init__(args, deo_args, W)

    def init_state(self, n):
        return np.repeat(np.atleast_2d(np.hstack((
            np.concatenate((
                np.repeat(1/modes, modes),
                np.zeros(modes),
                5 * np.ones(modes),
                np.zeros(data_len))),
        ))),
            n, axis=0)

experiment = Experiment_mixture(arg, deo_args, W_mixture)
experiment.run()
