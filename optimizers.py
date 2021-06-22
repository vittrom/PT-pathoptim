import autograd.numpy as np

class Adam():

    def __init__(self, alpha=0.001, eps=10e-8, b1=0.9, b2=0.999):
        self.m = 0
        self.v = 0
        self.alpha = alpha
        self.eps = eps
        self.b1 = b1
        self.b2 = b2
        self.current_iter = 0

    def update(self, beta, gradient):
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * gradient ** 2
        m_hat = self.m / (1 - self.b1 ** (self.current_iter + 1))
        v_hat = self.v / (1 - self.b2 ** (self.current_iter + 1))

        self.current_iter += 1

        return beta - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)

class Saga():

    def __init__(self, gamma=0.01, prox=lambda x, gamma: x):
        self.gamma = gamma
        self.prox = prox

    def update(self, beta, gradient, prev_grad, avg_grad):
        mean_grad = np.nanmean(avg_grad, axis=0)
        res = beta - self.gamma * (gradient - prev_grad + mean_grad)  # sign inverted

        return self.prox(res, self.gamma)

class SGD():

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, beta, gradient, args=None):
        return beta - self.lr * gradient

class SGD_momentum():
    def __init__(self, lr=0.01, grad_w=0.9, ma_length=10):
        self.lr = lr
        self.calls = 0
        self.ma_length = ma_length
        self.grads = [np.nan] * ma_length
        self.ma = 0
        self.grad_w = grad_w

    def update_ma(self):
        if self.calls == 1:
            self.ma = self.grads[0]
        else:
            grads = self.grads[0:self.calls]
            self.ma = np.mean(np.array(grads), 0)

    def update(self, beta, gradient, args=None):
        if self.calls > 0:
            self.update_ma()
        self.grads[self.calls] = gradient
        if self.calls >= self.ma_length - 1:
            self.calls = 0
        else:
            self.calls += 1

        g = self.grad_w * gradient + (1 - self.grad_w) * self.ma

        return beta - self.lr * g

class Adagrad():
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = 0
        self.calls = 0

    def update_G(self, gradient):
        if self.calls == 0:
            self.G = gradient ** 2
        else:
            self.G += gradient ** 2

    def update(self, beta, gradient, args=None):
        self.update_G(gradient=gradient)
        self.calls += 1
        return beta - self.lr * gradient / (np.sqrt(self.G + self.eps))