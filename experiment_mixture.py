import os
from PT import DEO
from optimizers import Adagrad
import argparse
from W_mixture import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
parser.add_argument('--datapath', type=str, default=os.path.join("experiments_data", "galaxies.csv"),
                    help="Path to data")
parser.add_argument('--modes', type=int, default=6, # For 1 segment use 2 segments with optim False
                    help="Number of modes")
parser.add_argument('--segments', type=int, default=3, # For 1 segment use 2 segments with optim False
                    help="Number of knots in spline")
arg = parser.parse_args()

# Create folders for saving if not existing
if not os.path.exists(arg.foldername):
    os.makedirs(arg.foldername)
if not os.path.exists(os.path.join(arg.foldername, "path")):
    os.makedirs(os.path.join(arg.foldername, "path"))
if not os.path.exists(os.path.join(arg.foldername, "path_log")):
    os.makedirs(os.path.join(arg.foldername, "path_log"))

# Load data
data = np.genfromtxt(arg.datapath, skip_header=-1, delimiter=",")
N = arg.N
modes = arg.modes
#true modes = data generation
# 50 points of N(100, 5^2)
# 150 points of N(200, 10^2)
# so w approx (.25, .75)
# mu (100, 200)
# sd (5, 10)

# args = [[mu0, sd0], [mu1, sd1], [a0, b0], [a1, b1], data] (a0/1, b0/1 are shape/scale for inverse gamma)
args = [modes] + modes * [[150., 1.]] + modes * [[1.1, 10]] + [data]

optim_iters = arg.optim_steps
K = arg.K
n_repl = arg.n_repl
iters = optim_iters * K
tune = arg.tune
reversible = arg.reversible
segments = arg.segments


total_time = []
# Same computational budget but all on schedule, do 2*n every step
if tune and not arg.optim:
     optim_iters = int(np.log2(iters + 1) - 1)
     K = 2

for n in N:
    eta = np.concatenate(((1 - np.arange(1, segments) / segments).reshape(-1, 1),
                          (np.arange(1, segments) / segments).reshape(-1, 1)), axis=1)

    eta = np.log(eta)
    beta = np.arange(n) / (n - 1)

    for i in range(n_repl):
        current_state = np.repeat(np.atleast_2d(np.hstack((
            np.concatenate((
                np.repeat(1/modes, modes),
                np.zeros(modes),
                5 * np.ones(modes),
                np.zeros(data.shape[0]))),
        ))),
            beta.shape[0], axis=0)
        pt = DEO(kernels=kernels, n_expl=1,
                 W=W_eta_segments(segments), W_KL=loss_KL_segments(segments),
                 gradient_fn=gradient_cov_segments(segments),
                 phi=eta, args=args, optimizer=Adagrad(lr=arg.lr), annealing_pars=beta)
        pt.simulate(iters=iters, init_state=current_state, burn_in=100, K=K,
                    KL=True, optim=arg.optim, optim_iters=optim_iters,
                    savepath=arg.foldername,
                    filename="results__n" + str(n) + "_repl_" + str(i),
                    tune=tune, reversible=reversible)

