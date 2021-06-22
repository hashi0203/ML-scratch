# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as rd

# データの取得
iris = load_iris()
x = iris.data

# PCA による二次元への圧縮
datas = len(x)
factors = len(x[0])
C = np.zeros((factors,factors))
for i in range(datas):
    tmp = x[i].reshape(factors,1)
    C += tmp.dot(tmp.T)
_, e = np.linalg.eigh(C)
T = e[:, :-3:-1]
compressed = x.dot(T)

plt.figure()
colors = ['violet','skyblue','yellow']
cmap = ListedColormap(colors)
plt.scatter(compressed.T[0],compressed.T[1],c=iris.target,cmap=cmap,alpha=0.5)
plt.savefig("compressed.png")

# k 平均クラスタリング
c = 3
mu = np.array([compressed[rd.randrange(datas)] for _ in range(c)])
y = np.array([rd.randrange(c) for _ in range(datas)])

def add(x,y,c):
    ans = np.array([0.0 for _ in range(2)])
    for i in range(datas):
        if y[i] == c:
            ans += x[i]
    return ans

while True:
    new_y = np.array([np.argmin(np.array([np.linalg.norm(compressed[i]-mu[j],ord=2) for j in range(c)])) for i in range(datas)])
    nc = np.array([np.count_nonzero(new_y==i) for i in range(c)])
    new_mu = np.array([1.0/nc[i]*add(compressed,new_y,i) for i in range(c)])
    if (new_y == y).all() and (new_mu == mu).all():
        y = new_y
        mu = new_mu
        break
    y = new_y
    mu = new_mu

plt.figure()
plt.scatter(compressed.T[0],compressed.T[1],c=y,cmap=cmap,alpha=0.5)
colors = ['red','blue','gold']
cmap = ListedColormap(colors)
plt.scatter(mu.T[0],mu.T[1],marker='x',c=np.array(range(c)),cmap=cmap)
plt.savefig("clustered.png")
