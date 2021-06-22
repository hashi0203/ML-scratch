import numpy as np
from copy import deepcopy
import csv
import os

def update_rho(N,mu,V):
    rho = 1/N * (np.diag(np.sum(V, axis=0)) + np.sum(mu**2, axis=0))
    return rho

def update_sigma(R,O,muu,muv,Vu,Vv):
    rr = sum([R[j][i]**2 for (j,i) in O])
    rmumu = sum([R[j][i] * muu[j].dot(muv[i]) for (j,i) in O])
    tr = sum([np.trace((Vu[j] + muu[j].reshape(-1,1).dot(muu[j].reshape(1,-1))).dot(Vv[i] + muv[i].reshape(-1,1).dot(muv[i].reshape(1,-1)))) for (j,i) in O])
    sigma = 1/(len(O)) * (rr - 2*rmumu + tr)
    return sigma

def update_user(J,R,rhou,sigma,muv,Vv):
    Vu = sigma * np.array([np.linalg.inv(np.sum(np.array([Vv[i] + muv[i].reshape(-1,1).dot(muv[i].reshape(1,-1)) for i in Aj[j]]), axis=0) + sigma*np.diag(1/rhou)) for j in range(J)])
    muu = 1/sigma * np.array([Vu[j].dot(np.array([0 for k in range(K)]) + np.sum([R[j][i] * muv[i] for i in Aj[j]], axis=0)) for j in range(J)])
    return muu, Vu

def update_item(I,R,rhov,sigma,muu,Vu):
    Vv = sigma * np.array([np.linalg.inv(np.sum(np.array([Vu[j] + muu[j].reshape(-1,1).dot(muu[j].reshape(1,-1)) for j in Ai[i]]), axis=0) + sigma*np.diag(1/rhov)) for i in range(I)])
    muv = 1/sigma * np.array([Vv[i].dot(np.array([0 for k in range(K)]) + np.sum([R[j][i] * muu[j] for j in Ai[i]], axis=0)) for i in range(I)])
    return muv, Vv

# 小さい行列で処理 (あまりうまくいかない)
print("------ Small Matrix ------")
R = np.array([[3,3,0,1],[3,0,3,0],[1,0,0,3],[0,3,3,0],[0,0,1,3]], dtype=float)
J = len(R)
I = len(R[0])
O = np.array([(j,i) for j in range(J) for i in range(I) if R[j][i] != 0])
Aj = [np.where(R[j]!=0)[0] for j in range(J)]
Ai = [np.where(R[:,i]!=0)[0] for i in range(I)]
K = 2

Vv = np.array([[[1.0 if k1 == k2 else 0.0 for k1 in range(K)] for k2 in range(K)] for _ in range(I)], dtype=float)
muv = np.array([[1.0 if k == i else 0.0 for k in range(K)] for i in range(I)], dtype=float)

muv = np.abs(np.random.normal(0,2,(I,K)))
Vv = np.abs(np.random.normal(0,1,(I,K,K)))
rhou = np.array([1/K for _ in range(K)], dtype=float)
rhov = np.array([1.0 for _ in range(K)], dtype=float)
sigma = 1

R1 = np.array([[1.0 for _ in range(I)] for _ in range(J)], dtype=float)
R2 = np.array([[0.0 for _ in range(I)] for _ in range(J)], dtype=float)
while np.max(np.abs(R1 - R2)) > 0.01:
    muu,Vu = update_user(J,R,rhou,sigma,muv,Vv)
    muv,Vv = update_item(I,R,rhov,sigma,muu,Vu)
    R1 = deepcopy(R2)
    R2 = muu.dot(muv.T)

    rhou = update_rho(I,muu,Vu)
    rhov = update_rho(I,muv,Vv)
    sigma = update_sigma(R,O,muu,muv,Vu,Vv)
    print('errors (max(R_t-R_(t-1))): {}'.format(np.max(np.abs(R1 - R2))))

print('Result matrix:')
print(R2)

ans = deepcopy(R2)
for (j,i) in O:
    ans[j][i] = R[j][i]
print('Answer matrix:')
print(ans)

diff =R2-ans
print('Diff matrix (Result-Answer):')
print(diff)

print('Average diff (ave(diff)):')
print(np.sum(np.abs(diff))/len(O))

print()

# 大きめの行列で処理 (小さい行列よりは良い結果) (R^T | R^T)^T
print("------ Larger Matrix ------")
R = np.array([[3,3,0,1],[3,0,3,0],[1,0,0,3],[0,3,3,0],[0,0,1,3]], dtype=float)
R = np.vstack([R,R])
J = len(R)
I = len(R[0])
O = np.array([(j,i) for j in range(J) for i in range(I) if R[j][i] != 0])
Aj = [np.where(R[j]!=0)[0] for j in range(J)]
Ai = [np.where(R[:,i]!=0)[0] for i in range(I)]
K = 2

Vv = np.array([[[1.0 if k1 == k2 else 0.0 for k1 in range(K)] for k2 in range(K)] for _ in range(I)], dtype=float)
muv = np.array([[1.0 if k == i else 0.0 for k in range(K)] for i in range(I)], dtype=float)

muv = np.abs(np.random.normal(0,2,(I,K)))
Vv = np.abs(np.random.normal(0,1,(I,K,K)))
rhou = np.array([1/K for _ in range(K)], dtype=float)
rhov = np.array([1.0 for _ in range(K)], dtype=float)
sigma = 1

R1 = np.array([[1.0 for _ in range(I)] for _ in range(J)], dtype=float)
R2 = np.array([[0.0 for _ in range(I)] for _ in range(J)], dtype=float)
while np.max(np.abs(R1 - R2)) > 0.01:
    muu,Vu = update_user(J,R,rhou,sigma,muv,Vv)
    muv,Vv = update_item(I,R,rhov,sigma,muu,Vu)
    R1 = deepcopy(R2)
    R2 = muu.dot(muv.T)

    rhou = update_rho(I,muu,Vu)
    rhov = update_rho(I,muv,Vv)
    sigma = update_sigma(R,O,muu,muv,Vu,Vv)
    print('errors (max(R_t-R_(t-1))): {}'.format(np.max(np.abs(R1 - R2))))

print('Result matrix:')
print(R2)

ans = deepcopy(R2)
for (j,i) in O:
    ans[j][i] = R[j][i]
print('Answer matrix:')
print(ans)

diff =R2-ans
print('Diff matrix (Result-Answer):')
print(diff)

print('Average diff (ave(diff)):')
print(np.sum(np.abs(diff))/len(O))

print()

# 映画の推薦用の大きなデータ (ある程度うまくいっている)
print("------ Large Matrix ------")
# load data
with open('ratings.csv') as f:
    reader = csv.reader(f)
    data = [row[:3] for row in reader]
data = np.array(data[1:],dtype=float)
data[:,2] *= 2
data = data.astype(int)
print('Initial data is in \"ratings.csv\"')
# print(data)
J = np.max(data[:,0])
I = np.max(data[:,1])
min_score = np.min(data[:,2])
max_score = np.max(data[:,2])

# data processing
R = np.zeros((J,I), dtype=float)
for (j,i,s) in data:
    R[j-1][i-1] = s
# remove row and column whose values of all cells are zero
R = np.delete(R,np.where(np.sum(R,axis=0)==0),1)
R = np.delete(R,np.where(np.sum(R,axis=1)==0),0)
J = len(R)
I = len(R[0])
print('num of people: {}, num of movies: {}'.format(J,I))
# observation list
O = np.array([(j,i) for j in range(J) for i in range(I) if R[j][i] != 0])
# adjacent list
Aj = [np.where(R[j]!=0)[0] for j in range(J)]
Ai = [np.where(R[:,i]!=0)[0] for i in range(I)]
K = 5

if not os.path.exists('output'):
    os.mkdir('output')
with open('output/ratings_init.csv'.format(i), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(R)
print('Initial matrix is saved in \"output/ratings_init.csv\"')
# print(R)

# initialize
Vv = np.array([[[1.0 if k1 == k2 else 0.0 for k1 in range(K)] for k2 in range(K)] for _ in range(I)], dtype=float)
muv = np.array([[1.0 if k == i else 0.0 for k in range(K)] for i in range(I)], dtype=float)
rhou = np.array([1/K for _ in range(K)], dtype=float)
rhov = np.array([1.0 for _ in range(K)], dtype=float)
sigma = 1

# dummy matrixs to store prior result
R1 = np.array([[1.0 for _ in range(I)] for _ in range(J)], dtype=float)
R2 = np.array([[0.0 for _ in range(I)] for _ in range(J)], dtype=float)
while np.max(np.abs(R1 - R2)) > 0.01:
    muu,Vu = update_user(J,R,rhou,sigma,muv,Vv)
    muv,Vv = update_item(I,R,rhov,sigma,muu,Vu)
    R1 = deepcopy(R2)
    R2 = muu.dot(muv.T)

    rhou = update_rho(I,muu,Vu)
    rhov = update_rho(I,muv,Vv)
    sigma = update_sigma(R,O,muu,muv,Vu,Vv)
    print('errors (max(R_t-R_(t-1))): {}'.format(np.max(np.abs(R1 - R2))))

print('Result matrix is saved in \"output/ratings_result.csv\"')
# print(R2)
with open('output/ratings_result.csv'.format(i), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(R2)

ans = deepcopy(R2)
for (j,i) in O:
    ans[j][i] = R[j][i]
print('Answer matrix is saved in \"output/ratings_ans.csv\"')
# print(ans)
with open('output/ratings_ans.csv'.format(i), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(ans)

diff =R2-ans
print('Diff matrix (Result-Answer) is saved in \"output/ratings_diff.csv\"')
# print(diff)
with open('output/ratings_diff.csv'.format(i), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(diff)

print('Average diff (ave(diff)):')
print(np.sum(np.abs(diff))/len(O))


