import numpy as np
import pandas as pd

def generate_mixture(weights, means, sds, N, filename, dims=1):
    u = np.random.rand(N)
    prev_w = 0
    x = []
    for i in range(len(weights)):
        n = ((u > prev_w) & (u <= prev_w + weights[i])).sum()
        prev_w += weights[i]
        x.append(means[i] + sds[i]*np.random.normal(size=(n, dims)))

    data = np.vstack(x).squeeze()
    pd.DataFrame(data).to_csv(filename, index=False, header=None)



generate_mixture(weights=[0.3, 0.7], means=[100., 200.], sds=[10., 10.], N=1000, dims=3,
                 filename="multivariate_mix.csv")

