import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(sample_size=90, n_class=3):
     x = (np.random.normal(size=(sample_size // n_class, n_class))
          + np.linspace(-3., 3., n_class)).flatten()
     y = np.broadcast_to(np.arange(n_class),
                         (sample_size // n_class, n_class)).flatten()
     return x, y

def build_design_mat(x1, x2, h):
    return np.exp(-(x1[:, None] - x2[None]) ** 2 / (2 * h ** 2))

def visualize(theta, x, y, h):
     xmin = -5
     xmax = 5
     n_class = len(theta)
     X = np.linspace(-5, 5, 1000)
     phi = [build_design_mat(x[y==i], X, h) for i in range(n_class)]
     tp = np.array([theta[i].dot(phi[i]) for i in range(n_class)])
     tp = np.where(tp > 0, tp, 0)
     s = np.sum(tp, axis=0)
     p = tp / s[np.newaxis, :]
     plt.clf()
     plt.grid()
     plt.rcParams['axes.axisbelow'] = True
     plt.xlim(xmin-0.3, xmax+0.3)
     plt.xticks(range(xmin, xmax+1))
     colors = ['blue', 'red', 'green']
     lines = ['solid', 'dashed', 'dashdot']
     markers = ['o', 'x', 'v']
     for i in range(n_class):
          plt.plot(X, p[i], color=colors[i], linestyle=lines[i], label='p(y='+str(i+1)+'|x)')
          xx = x[y==i]
          plt.scatter(xx, np.zeros_like(xx)-(abs(i-1)+1)/20, c=colors[i], marker=markers[i])
     plt.legend()
     plt.savefig('output.png')

sample_size = 90
n_class = 3
h = 1
l = 1
x, y = generate_data(sample_size, n_class)
phi = [build_design_mat(x, x[y==i], h) for i in range(n_class)]
pi = [np.where(y == i, 1, 0) for i in range(n_class)]
theta = np.array([np.linalg.solve(
     phi[i].T.dot(phi[i]) + l * np.eye(phi[i].shape[1]), phi[i].T.dot(pi[i])
) for i in range(n_class)])
visualize(theta, x, y, h)
