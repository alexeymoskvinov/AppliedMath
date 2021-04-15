
import ConjugateGradientMethod
import numpy as np
from scipy.sparse.linalg import cg
import scipy.optimize as scm
from scipy import optimize

n = 2
P = np.random.normal(size=[n, n])
A = np.dot(P.T, P)
print(A)
b = np.ones(n)
print(b)

x = ConjugateGradientMethod.conjugate_grad(A, b)
#x2 = np.linalg.solve(A, b)
x3 = cg(A, b)
#print(np.dot(A, x) - b)
print(x)
#print(x2)
print(x3)

def f(x):
    return x**2


#minimum = optimize.fmin_powell(f, -1)
#print(minimum)