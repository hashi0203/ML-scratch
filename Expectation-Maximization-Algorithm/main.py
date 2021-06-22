import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

np.random.seed(0)
def data_generate(n):
    return (np.random.randn(n) + np.where(np.random.rand(n) > 0.3, 2., -2.))

def visualize(x, params, m):
    plt.figure()
    plt.xlim(-6,6)
    plt.hist(x, bins=20, color='k', alpha=0.4, density=True)
    X = np.linspace(-6,6,50)
    colorlist = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    if len(params) >= len(colorlist):
        q = len(params)//len(colorlist)
        r = len(params)%len(colorlist)
    else:
        q = 1
        r = 0
    for i in range(len(colorlist)):
        idx = q*(i+1)+r-1
        if idx >= len(params):
            break
        w,mu,sigma = params[idx]
        phi = np.sum(np.array([w[j]*norm.pdf(X,mu[j],sigma[j]) for j in range(m)]), axis=0)
        plt.plot(X,phi,color=colorlist[i%len(colorlist)],label=idx)
    plt.legend()
    plt.title("Mixed number: %d" % m)
    plt.savefig("output-%d" % m)

def calc_eta(x,w,mu,sigma):
    w_phi = np.array([[w[j]*norm.pdf(x[i],mu[j],sigma[j]) for j in range(m)] for i in range(n)])
    return np.array([l/np.sum(l) for l in w_phi])

def em_algo(x,w,mu,sigma):
    eta = calc_eta(x,w,mu,sigma)
    eta_sumi = np.sum(eta, axis=0)
    new_w = eta_sumi/n
    new_mu = x.dot(eta)/eta_sumi
    new_sigma = np.array([np.sqrt(((x-mu[j])**2).dot(eta[:, j])/eta_sumi[j]) for j in range(m)])
    if np.linalg.norm(np.append(new_w-w,np.append(new_mu-mu,new_sigma-sigma))) < 10e-2:
        return np.append(np.array([[w,mu,sigma]]),np.array([[new_w,new_mu,new_sigma]]),axis=0)
    else:
        return np.append(np.array([[w,mu,sigma]]),em_algo(x,new_w,new_mu,new_sigma),axis=0)

n = 1000
ms = [10, 5]

for m in ms:
    x = np.array(data_generate(n))
    w,mu,sigma = np.ones(m)/m,np.linspace(1,6,m),np.linspace(1,6,m)
    print("初期値")
    print("w:",w)
    print("mu:",mu)
    print("sigma:",sigma)
    start = time.time()
    params = em_algo(x,w,mu,sigma)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("反復回数:",len(params))
    print("推定値")
    print("w:",params[-1][0])
    print("mu:",params[-1][1])
    print("sigma:",params[-1][2])
    visualize(x,params,m)


