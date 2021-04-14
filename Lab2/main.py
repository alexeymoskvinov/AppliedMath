
import ConjugateGradientMethod
import numpy as np
from scipy.sparse.linalg import cg


n = 1000
P = np.random.normal(size=[n, n])
A = np.dot(P.T, P)
b = np.ones(n)


x = ConjugateGradientMethod.conjugate_grad(A, b)
x2 = np.linalg.solve(A, b)
x3 = cg(A, b)
print(np.dot(A, x) - b)