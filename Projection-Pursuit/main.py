import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

def generate_data(n=1000):
    x = np.concatenate([np.random.rand(n, 1), np.random.randn(n, 1)], axis=1)
    x[0, 1] = 6   # outlier
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Standardization
    M = np.array([[1, 3], [5, 3]])
    x = x.dot(M.T)
    x = np.linalg.inv(sqrtm(np.cov(x, rowvar=False))).dot(x.T).T

    return x

def newton(n, b, x, g, dg):
    bx = b.T.dot(x.T)
    newb = 1/n*np.sum(dg(bx))*b - 1/n*np.sum(g(bx)[:, np.newaxis]*x, axis=0)
    # newb の1つ目の数の値を0以上にする
    newb = np.sign(np.sign(newb[0])+0.5)/np.linalg.norm(newb)*newb
    if np.linalg.norm(newb - b) < 10e-3:
        return np.array([b,newb])
    else:
        return np.append([b], newton(n, newb, x, g, dg), axis=0)

def visualize(x, bs, axes0, axes1, title):
    axes0.set_title(title)
    axes0.set_xlim(-6,6)
    axes0.set_ylim(-4,4)
    [X,Y] = x.T
    axes0.scatter(X,Y, s = 15, marker='x', color='red')
    X = np.linspace(-6,6,10)
    colorlist = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    for i in range(len(bs)):
        if bs[i][0] == 0:
            axes0.plot([0,0],[-4,4], color=colorlist[i%len(colorlist)], label=i)
        else:
            axes0.plot(X,bs[i][1]/bs[i][0]*X, color=colorlist[i%len(colorlist)], label=i)
    axes0.legend()
    axes0.set_aspect('equal')

    axes1.set_title(title)
    axes1.set_xlim(-6,6)
    # 得られたベクトルの方向に正射影したときの距離
    proj = bs[-1].dot(x.T)/np.linalg.norm(bs[-1])
    axes1.hist(proj, color='purple')

n = 1000
b = np.ones(2)
x = generate_data(n)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

g = lambda s : 4*s**3
dg = lambda s : 12*s**2
bs = newton(n, b, x, g, dg)
print('(%.2f, %.2f)' % tuple(bs[-1]))
visualize(x, bs, axes[0, 0], axes[0, 1], "$G(s) = s^4$")

g = lambda s : np.tanh(s)
dg = lambda s : 1 - np.tanh(s)**2
bs = newton(n, b, x, g, dg)
print('(%.2f, %.2f)' % tuple(bs[-1]))
visualize(x, bs, axes[1, 0], axes[1, 1], "$G(s) = \log{(\cosh{(s)})}$")

g = lambda s : s*np.exp(-s**2/2)
dg = lambda s : (1-s**2)*np.exp(-s**2/2)
bs = newton(n, b, x, g, dg)
print('(%.2f, %.2f)' % tuple(bs[-1]))
visualize(x, bs, axes[2, 0], axes[2, 1], "$G(s) = -\exp{(-\dfrac{s^2}{2})}$")

fig.tight_layout()
plt.savefig("output.png")
