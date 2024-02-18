import autograd.numpy as np
from autograd import elementwise_grad, jacobian
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import copy
import sys

class Covariance():
    """
    Class to calculate covariance.
    """

    def __init__(self):
        """
        Initialize the Covariance object.
        """
        self.meanx = 0
        self.meany = 0
        self.C = 0
        self.n = 0

    def reset_cov(self):
        """
        Reset the covariance values.
        """
        self.meanx = 0
        self.meany = 0
        self.C = 0
        self.n = 0

    def update_cov(self, x, y):
        """
        Update the covariance based on new x and y values.

        Args:
            x: The x value.
            y: The y value.
        """
        self.n += 1
        dx = x - self.meanx
        self.meanx += dx / self.n
        self.meany += (y - self.meany) / self.n
        self.C += dx * (y - self.meany)

    def sample_covariance(self):
        """
        Calculate the sample covariance.

        Returns:
            The sample covariance.
        """
        return self.C / (self.n - 1)

    def pop_covariance(self):
        """
        Calculate the population covariance.

        Returns:
            The population covariance.
        """
        return self.C / self.n

# Functions for gradient calculation
def compute_log_ratio(W, X, phi, betas, args):
    return -W(phi, betas[0:-1], X=X[1::], args=args) - W(phi, betas[1::], X=X[0:-1], args=args) + W(phi, betas[0:-1], X=X[0:-1], args=args) + W(phi, betas[1::], X=X[1::], args=args)

def swap_back(current_state, is_swapped, N):
    r = np.arange(N - 1)
    r_s = r[is_swapped == True]
    temp = copy.deepcopy(current_state)
    tmp = temp[r_s]
    temp[r_s] = temp[r_s + 1]
    temp[r_s + 1] = tmp

    return temp

def S(W, phi, x, betas, args):
    fn = jacobian(W)
    grad_w_i_x = fn(phi, betas, X=x, args=args)
    return grad_w_i_x

def S_KL(W, phi, x, betas, args):
    fn = elementwise_grad(W)
    grad_w_i_x = fn(phi, betas, X=x, args=args)
    return grad_w_i_x

#Additional functions
def plot_results(data, save_path, x_label, y_label, title, window=None, fns=None, dimensions=None):
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    if dimensions is None:
        dimensions = [-1]
    data_size =data.shape[0]
    if fns is None:
        for i in dimensions:
            plt.plot(data[:, i])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        plt.savefig(save_path)
        plt.close()
    else:
        for k in range(len(fns)):
            #Compute function for window steps
            fn_res = []
            for i in np.linspace(window, data_size, data_size/window):
                fn_res.append(fns[k](data[0:int(i), dimensions], axis=0))
            fn_res = np.array(fn_res)
            for i in dimensions:
                plt.plot(fn_res[:, i], label=str(i))

            plt.xlabel(x_label[k])
            plt.ylabel(y_label[k])
            plt.title(title[k])
            plt.legend()
            plt.savefig(save_path[k])
            plt.close()

def bisection(f, a, b, tol = 1e-10, maxiter = 3000):
    fa = f(a)
    if fa * f(b) <= 0:
        pass
    else:
        raise Exception("No real root in [a,b]")
    i = 0
    c = 0
    while b-a > tol or i < maxiter:
        i += 1
        c = (a+b)/2
        fc = f(c)
        if fc == 0:
            break
        elif fa*fc > 0:
            a = c  # Root is in the right half of [a,b].
            fa = fc
        else:
            b = c  # Root is in the left half of [a,b].
    return c

def make_plots_normal(update_fn, save_path):
    mu, sd = update_fn()
    # mu, sd = update_fn(alpha(betas, pars), gamma(betas, pars), args)
    tau = 1 / (sd ** 2)
    plt.scatter(mu, tau)
    plt.xlabel("mu")
    plt.ylabel("precision")
    plt.savefig(save_path)
    plt.close()

def plot_etas_log(etas, knots, args, args_names, savepath=None, extra_pars=0):
    # args = [GCB, SKL, mean_rej, rejections]
    if extra_pars != 0:
        etas = etas[:-extra_pars].reshape(-1, 2)
    rejections = args[-1]
    textstr = ""
    for k in range(len(args_names)):
        textstr += args_names[k] + ": " + str(np.round(args[k], 2))
        if k == len(args_names) - 1:
            pass
        else:
            textstr += ", "
    breaks = np.concatenate((np.array([1, 0]).reshape(-1, 2), knots, np.array([0, 1]).reshape(-1, 2)), axis=0)

    for i in range(breaks.shape[0] - 1):
        x_coord = np.linspace(breaks[i, 0], breaks[i + 1, 0], 1000)
        y_coord = np.linspace(breaks[i, 1], breaks[i + 1, 1], 1000)
        log_x = np.log(np.maximum(1e-200, x_coord))
        log_x[log_x < -400] = -np.inf
        log_y = np.log(np.maximum(1e-200, y_coord))
        log_y[log_y < -400] = -np.inf
        plt.plot(log_x, log_y, zorder=1)

    etas = np.log(np.maximum(1e-200, etas))
    etas[etas < -400] = -np.inf
    plt.scatter(etas[:, 0], etas[:, 1], c=np.concatenate((rejections, np.array([0]))),
                vmin=0.0, vmax=1.0, zorder=2)
    # plt.plot(breaks[:, 0], breaks[:, 1])
    plt.title(textstr)
    plt.colorbar()
    plt.xlabel("Eta_0")
    plt.ylabel("Eta_1")
    plt.savefig(savepath)
    plt.close()

def plot_etas(etas, knots, args, args_names, savepath=None, extra_pars=0):
    if extra_pars != 0:
        etas = etas[:-extra_pars].reshape(-1, 2)
    # args = [GCB, SKL, mean_rej, rejections]
    rejections = args[-1]
    textstr = ""
    for k in range(len(args_names)):
        textstr += args_names[k] + ": " + str(np.round(args[k], 2))
        if k == len(args_names) - 1:
            pass
        else:
            textstr += ", "
    breaks = np.concatenate((np.array([1, 0]).reshape(-1, 2), knots, np.array([0, 1]).reshape(-1, 2)), axis=0)

    for i in range(breaks.shape[0] - 1):
        x_coord = np.linspace(breaks[i, 0], breaks[i + 1, 0], 1000)
        y_coord = np.linspace(breaks[i, 1], breaks[i + 1, 1], 1000)
        plt.plot(x_coord, y_coord, zorder=1)

    plt.scatter(etas[:, 0], etas[:, 1], c=np.concatenate((rejections, np.array([0]))),
                vmin=0.0, vmax=1.0, zorder=2)
    plt.title(textstr)
    plt.colorbar()
    plt.xlabel("Eta_0")
    plt.ylabel("Eta_1")
    plt.savefig(savepath)
    plt.close()

def fix_monotonicity(phi, decreasing=True):
    if len(phi) == 1:
        return phi
    else:
        # First dimension needs to be decreasing
        x = phi
        # Find indexes that have non monotonicity
        non_monotonic = []
        non_mono = False
        repeat = True
        while repeat:
            for i in range(len(x) - 1):
                cond = False
                if decreasing:
                    if x[i] >= x[i + 1]:
                        non_mono = False
                        cond = True
                else:
                    if x[i] <= x[i + 1]:
                        non_mono = False
                        cond = True

                if not cond:
                    if non_mono:
                        tmp = non_monotonic[-1] + [i + 1]
                        non_monotonic.pop(-1)
                        non_monotonic.append(tmp)
                    else:
                        non_monotonic.append([i,  i+1])
                        non_mono = True
            # Fix for last element
            if decreasing:
                if x[-1] > x[-2]: #increasing in the last element
                    if non_mono:
                        tmp = non_monotonic[-1] + [len(x) - 1]
                        non_monotonic.pop(-1)
                        non_monotonic.append(tmp)
                    else:
                        non_monotonic.append([len(x) - 1])
            else:
                if x[-1] < x[-2]:
                    if non_mono:
                        tmp = non_monotonic[-1] + [len(x) - 1]
                        non_monotonic.pop(-1)
                        non_monotonic.append(tmp)
                    else:
                        non_monotonic.append([len(x) - 1])

            if len(non_monotonic) > 0:
                for l in non_monotonic:
                    start = l[0]
                    end = l[-1]
                    if start == 0:
                        if decreasing:
                            prev = 0
                        else:
                            prev = -sys.maxsize
                    else:
                        prev = x[start - 1]
                    if end == len(phi) - 1:
                        if decreasing:
                            next = -sys.maxsize
                        else:
                            next = 0
                    else:
                        next = x[end + 1]

                    len_segment = np.abs(next - prev)
                    if decreasing:
                        x[l] = next + len_segment *  np.flip(np.arange(1, len(l) + 1))/(len(l) + 1)
                    else:
                        x[l] = prev + len_segment * np.arange(1, len(l) + 1)/(len(l) + 1)
                    non_monotonic = []
            else:
                repeat = False

        return x

def fix_monotonicity_2(x):
    if x.shape[0] <= 2:
        return x
    # figure out the order from x[0] and x[-1]
    if x[-1] > x[0]:
        decreasing = False
    else:
        decreasing = True

    # identify the indices of a valid monotone subsequence
    valid = [0]
    curval = x[0]
    for i in range(1, x.shape[0]-1):
       #if decreasing: x must be less than curval and more than last val
       #if increasing: x must be more than curval and less than last val
       if (decreasing and x[i] <= curval and x[i] >= x[-1]) or (not decreasing and x[i] >= curval and x[i] <= x[-1]):
           valid.append(i)
           curval = x[i]
    valid.append(x.shape[0]-1)

    # fill in any missing indices with interpolation
    xp = np.zeros(x.shape[0])
    for i in range(len(valid)-1):
        for j in range(valid[i], valid[i+1]):
            xp[j] = ( (valid[i+1]-j)*x[valid[i]] + (j - valid[i])*x[valid[i+1]] )/(valid[i+1]-valid[i])
    xp[-1] = x[-1]

    return xp
