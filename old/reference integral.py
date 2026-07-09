import numpy as np
from scipy.optimize import brentq

N = 4000
k = np.linspace(-np.pi, np.pi, N, endpoint=False)
KX, KY = np.meshgrid(k, k)
eps = -2 * (np.cos(KX) + np.cos(KY))

target = 7/16
mu = brentq(lambda m: (eps < m).mean() - target, -4, 4)
e0 = 2 * (eps * (eps < mu)).mean()
print(mu, e0)