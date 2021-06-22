import os
from PT import DEO
from optimizers import Adagrad
import argparse
from W_normals import *


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
parser.add_argument('--N', nargs="+", type=int, default=[50],
                    help="Number of chains")
parser.add_argument('--K', type=int, default=150,  # [100, 1000, 5000, 10000, 25000, 50000, 100000],
                    help="Number of samples to use for gradient approximation")
parser.add_argument('--n_repl', type=int, default=1,
                    help="Number of replications")
parser.add_argument('--tune', type=str2bool, default=True,
                    help="Schedule tuning rounds")
parser.add_argument('--optim_steps', type=int, default=500,
                    help="Number of optimization steps")
parser.add_argument('--optim', type=str2bool, default=True,
                    help="Do gradient optimization")
parser.add_argument('--foldername', type=str, default=os.path.join("Results", "test_normals", "plots_etas"),
                    help="Name folder to save results")
parser.add_argument('--lr', type=float, default=0.2,
                    help="Learning rate")
parser.add_argument('--checkpoint_iter', type=int, default=100,
                    help="Number of iter before checkpoint")
parser.add_argument('--mu0', type=float, default=-1,
                    help="Mean pi0")
parser.add_argument('--mu1', type=float, default=1,
                    help="Mean pi1")
parser.add_argument('--sigma0', type=float, default=0.01,
                    help="Sigma pi0")
parser.add_argument('--sigma1', type=float, default=0.01,
                    help="Sigma pi1")
parser.add_argument('--segments', type=int, default=2, #Use segments 2 with optim false for 1 segment
                    help="Number of knots in spline")
arg = parser.parse_args()

# Create folders for saving if not existing
if not os.path.exists(arg.foldername):
    os.makedirs(arg.foldername)
if not  os.path.exists(os.path.join(arg.foldername, "path")):
    os.makedirs(os.path.join(arg.foldername, "path"))
if not os.path.exists(os.path.join(arg.foldername, "path_log")):
    os.makedirs(os.path.join(arg.foldername, "path_log"))

N = arg.N
X = None
args = [X, arg.mu0, arg.sigma0, arg.mu1, arg.sigma1]
optim_iters = arg.optim_steps
K = arg.K
n_repl = arg.n_repl
iters = optim_iters * K
tune = arg.tune
reversible = arg.reversible
segments = arg.segments

print("Optimize: " + str(arg.optim))
print("Tune: " + str(arg.tune))
print("Reversible " + str(arg.reversible))

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
        current_state = np.array([np.random.normal(size=1) for j in range(n)])

        pt = DEO(kernels=kernels,
                 W=W_eta_segments(segments), W_KL=loss_KL_segments(segments),
                 gradient_fn=gradient_cov_segments(segments), n_expl=1,
                 phi=eta, args=args, optimizer=Adagrad(lr=arg.lr), annealing_pars=beta)
        pt.simulate(iters=iters, init_state=current_state, burn_in=0, K=K,
                    KL=True, optim=arg.optim, optim_iters=optim_iters,
                    savepath=arg.foldername,
                    filename="results__n" + str(n) + "_repl_" + str(i),
                    tune=tune, reversible=reversible)
