# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ステップ幅を固定して探索
def default(xl, yl, eps=0.08, max_itr=8):
    x1 = [xl]
    x2 = [yl]
    for _ in range(max_itr):
        xk = xl
        yk = yl
        # x, y の値を更新する
        xl = (1-20*eps)*xk
        yl = (1-2*eps)*yk
        x1.append(xl)
        x2.append(yl)
    return x1, x2

# バックトラック直線探索
def backtrack(xl, yl, a, b, max_itr=5):
    x1 = [xl]
    x2 = [yl]
    for _ in range(max_itr):
        xk = xl
        yk = yl
        # ek の上限を求める
        ek_max = (1-a)*(100*xk**2 + yk**2)/(1000*xk**2 + yk**2)
        # 上限以下になるまで b を掛けて ek を小さくする
        ek = 1.0
        while ek > ek_max:
            ek *= b
        # x, y の値を更新する
        xl = (1-20*ek)*xk
        yl = (1-2*ek)*yk
        x1.append(xl)
        x2.append(yl)
    return x1, x2

def visualize(x1, x2, name):
    plt.figure()
    plt.plot(x1, x2, marker='o')
    for i in range(len(x1)):
        plt.annotate(i, (x1[i], x2[i]+0.2))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    n = 100
    x = np.linspace(-5,5,n)
    y = np.linspace(-5,5,n)
    X, Y = np.meshgrid(x,y)
    Z = 10*X**2 + Y**2
    plt.contour(X,Y,Z,20)
    plt.gca().set_aspect('equal')
    plt.title(name)
    plt.savefig('output-%s.png' % name)

# 初期値設定
xl = 5.0
yl = 4.0
a = 0.5
b = 0.8

x1, x2 = default(xl, yl)
visualize(x1, x2, "default")

x1, x2 = backtrack(xl, yl, a, b)
visualize(x1, x2, "backtrack")
