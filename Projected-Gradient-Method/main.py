# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 射影勾配法
def pgm(mu0, eps, max_itr=10):
    nmu = mu0
    mu_ls = [mu0]
    x_ls = []
    y_ls = []
    for _ in range(max_itr):
        mu = nmu
        x = mu/6
        y = mu/4
        nmu = max(0, mu + eps*(-(5.0/12.0)*mu+1))
        mu_ls.append(nmu)
        x_ls.append(x)
        y_ls.append(y)
        if abs(nmu - mu) <= 0.1:
            break
    return mu_ls, x_ls, y_ls


# 初期値設定
mu0 = -2.0
eps0 = 1.0

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 18))

for i in range (5):
    eps = i + eps0
    mu_ls, x_ls, y_ls = pgm(mu0, eps)

    axes[i, 0].plot(x_ls,y_ls,marker='o')
    for t, (x, y) in enumerate(zip(x_ls, y_ls)):
        axes[i, 0].annotate(t+1, (x-0.03, y+0.03))
    axes[i, 0].set_xlabel('$x$')
    axes[i, 0].set_ylabel('$y$')
    n = 100
    x = np.linspace(-0.55,2.0,n)
    y = np.linspace(-0.55,2.0,n)
    X, Y = np.meshgrid(x,y)
    Z = 3*X**2 + 2*Y**2
    axes[i, 0].contour(X,Y,Z,20)
    axes[i, 0].set_aspect('equal')
    axes[i, 0].set_title('$\epsilon = %.1f$' % eps)

    time = range(len(mu_ls))
    axes[i, 1].plot(time,mu_ls,marker='o')
    axes[i, 1].set_xlabel('iteration')
    axes[i, 1].set_ylabel('$\mu$')
    axes[i, 1].set_title('$\epsilon = %.1f$' % eps)

fig.tight_layout()
plt.savefig("output.png")