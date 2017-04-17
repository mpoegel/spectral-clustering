from itertools import permutations
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time

start = time.time()
k = 5
m = 500
filename = 'data/raw/mnist'

n = 0
d = 0
with open(filename) as fp:
    for line in fp:
        bits = line.strip().split(' ')
        if int(bits[0]) < k:
            n += 1
        d = max(d, max([int(a.split(':')[0]) for a in bits[1:]]))
print(n, d)


def lineToVector(bits):
    i = 0
    k = 1
    x = np.empty(d)
    y = int(bits[0])
    while i < d and k < len(bits):
        col, val = bits[k].split(':')
        if int(col) == i:
            x[i] = float(val)
            k += 1
        else:
            x[i] = 0.0
        i += 1
    while i < d:
        x[i] = 0
        i += 1
    return x, y

# sample m columns uniformly at random without replacement
inds = np.random.choice(n, m, replace=False)
Z = np.empty((m, d))
Y = np.empty(n, dtype=int)
with open(filename) as fp:
    r = 0
    rr = 0
    for line in fp:
        bits = line.strip().split(' ')
        if int(bits[0]) < k:
            if r in inds:
                Z[rr], _ = lineToVector(bits)
                rr += 1
            Y[r] = int(bits[0])
            r += 1

# calculate the gaussian kernel parameter
mu = 0
for i in range(m):
    for j in range(m):
        mu += np.linalg.norm(Z[i] - Z[j]) ** 2
mu /= (m ** 2)
mu = 1 / mu
# created a sample of the affinity matrix
A_11 = np.empty((m, m))
for i in range(m):
    for j in range(i, m):
        val = np.e ** (-mu * np.linalg.norm(Z[i] - Z[j]) ** 2)
        A_11[i, j] = val
        A_11[j, i] = val

ww = A_11.dot(np.ones(m))
D_star = np.diag(ww)
D_star_ = np.diag(1 / np.sqrt(ww))
M_star = D_star_.dot(A_11).dot(D_star_)
# find the eigendecomposition of M_star
M_star = sp.cluster.vq.whiten(M_star)
Lam, V = sp.sparse.linalg.eigsh(M_star, k=k, which='LM')

Lam = np.diag(Lam)
B = D_star_.dot(V).dot(np.linalg.inv(Lam))

# create another affinity matrix row by row
Q = np.empty((n, k))
with open(filename) as fp:
    r = 0
    for line in fp:
        bits = line.strip().split(' ')
        if int(bits[0]) < k:
            x, _ = lineToVector(bits)
            a = np.array([np.linalg.norm(x - Z[j]) for j in range(m)])
            Q[r] = a.dot(B)
            r += 1

dd = Q.dot(Lam).dot(Q.T).dot(np.ones(n))
D_hat = np.diag(dd)
U = np.diag(1 / np.sqrt(dd)).dot(Q)
# orthogonalize U
P = U.T.dot(U)
Sig, Vp = sp.linalg.eigh(P)
Sig_ = np.diag(np.sqrt(Sig))
B = Sig_.dot(Vp.T).dot(Lam).dot(Vp).dot(Sig_)
Lam_tilde, V_tilde = sp.linalg.eigh(B)
U = U.dot(Vp).dot(np.diag(1 / np.sqrt(Sig))).dot(V_tilde)
# finally we have U as the approximate eigenvectors which we use to cluster
centroids, y_hat = sp.cluster.vq.kmeans2(U, k)

print('clustered eigenvectors in: {}'.format(time.time() - start))

accuracy = np.zeros(k)
perms = []
for p in permutations(np.arange(1, k + 1)):
    P = dict()
    for i in range(k):
        P[i] = p[i]
    perms.append(P)
accuracy = np.zeros(len(perms))
for i in range(len(perms)):
    yy = y_hat.copy()
    for key, val in perms[i].items():
        yy[y_hat == key] = val
    accuracy[i] = (Y+1 == yy).sum() / n * 100
print(accuracy.max())
