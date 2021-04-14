
import ConjugateGradientMethod
import numpy as np
from scipy.sparse.linalg import cg
import time

n = 1000
P = np.random.normal(size=[n, n])
A = np.dot(P.T, P)
b = np.ones(n)

t1 = time.time()
print('start')
x = ConjugateGradientMethod.conjugate_grad(A, b)
t2 = time.time()
print(t2 - t1)
x2 = np.linalg.solve(A, b)
t3 = time.time()
print(t3 - t2)
x3 = cg(A, b)
t4 = time.time()
print(t4 - t3)

print(np.dot(A, x) - b)