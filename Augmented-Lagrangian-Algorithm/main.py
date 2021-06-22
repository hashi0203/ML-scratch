# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 拡張ラグランジュ
def ala(l0, c, max_itr=20):
    nl = l0
    l_ls = [l0]
    x_ls = []
    y_ls = []
    # c = c0 + i * 2.0
    for _ in range(max_itr):
        l = nl
        x = (c - l)/(6+5*c/2)
        y = (c - l)/(4+5*c/3)
        nl = l + c*(x + y - 1.0)
        l_ls.append(nl)
        x_ls.append(x)
        y_ls.append(y)
        if abs(nl - l) <= 0.01:
            break
    return l_ls, x_ls, y_ls

# 初期値設定
l0 = 0.0
c0 = 2.0

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 18))

for i in range(5):
    c = i * 2 + c0
    l_ls, x_ls, y_ls = ala(l0, c)

    axes[i, 0].plot(x_ls,y_ls,marker='o')
    for t, (x, y) in enumerate(zip(x_ls, y_ls)):
        axes[i, 0].annotate(t+1, (x-0.03, y+0.03))
    axes[i, 0].set_xlabel('$x$')
    axes[i, 0].set_ylabel('$y$')
    n = 100
    x = np.linspace(-0.2,1.2,n)
    y = np.linspace(-0.2,1.2,n)
    X, Y = np.meshgrid(x,y)
    Z = 3*X**2 + 2*Y**2
    axes[i, 0].contour(X,Y,Z,20)
    axes[i, 0].set_aspect('equal')
    axes[i, 0].set_title('$c = %.1f$' % c)

    time = range(len(l_ls))
    axes[i, 1].plot(time,l_ls,marker='o')
    axes[i, 1].set_xlabel('iteration')
    axes[i, 1].set_ylabel('$\lambda$')
    axes[i, 1].set_title('$c = %.1f$' % c)

fig.tight_layout()
plt.savefig("output.png")
