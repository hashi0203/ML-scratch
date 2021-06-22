import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def data_generate(n):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1 / 8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1 / 8 <= u) & (u < 1 / 4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1 / 4 <= u) & (u < 1 / 2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1 / 2 <= u) & (u < 3 / 4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3 / 4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])

    return x

def visualize(x,X,hs,ps):
    plt.figure()
    plt.xlim(0,5)
    plt.title('Kernel Density Estimation')
    plt.hist(x, bins=20, color='r', alpha=0.4, density=True)
    colorlist = ['#377eb8', '#4daf4a', '#ff7f00']
    for i in range(len(hs)):
        plt.plot(X,ps[i],color=colorlist[i],label="h="+str(hs[i]), linewidth = 3.0)
    plt.legend()
    plt.savefig("output.png")

def kernel(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)

def p(X,x,n,h):
    return np.array([1/(n*h)*np.sum(kernel((xi-x)/h)) for xi in X])

def cross_validate(x,t,n,h):
    T = np.array_split(x, t)
    idx = 0
    LCV = 0
    for i in range(t):
        l = len(T[i])
        LCV += 1/l*np.sum(np.log(p(T[i],np.append(x[0:idx],x[idx+l:]),n,h)))
        idx += l
    return LCV/t

n = 3000
t = 5
x = data_generate(n)
hs = 1 / (2 ** np.arange(1, 9))
LCVs = np.array([cross_validate(x,t,n,h) for h in hs])
plt.figure()
plt.title('Cross Validation')
plt.grid()
plt.xlabel('h')
plt.plot(hs,LCVs)
plt.savefig("output-cv.png")

h = hs[np.argmax(LCVs)]
X = np.linspace(0,5,30)
visualize(x,X,[h,hs[0],hs[-1]],[p(X,x,n,h),p(X,x,n,hs[0]),p(X,x,n,hs[-1])])


