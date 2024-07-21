# using LinearAlgebra;
# using Plots;
# using LaTeXStrings;

# include("Cheby_LS.jl");
import cheby_LS as cheby
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

# parameters that can be played with
d = 100
# dimension
κ = 10**2 # condition number
maxiter = 500 # maximum number of iterations

σ = 1.
L = κ

# general problem instance
A = np.random.randn(d,d)
A = A.T @ A
lambdas, U = np.linalg.eig(A)
# lambdas = LinRange(σ, L, d)
lambdas = np.linspace(σ, L, d)
A = U @ np.diag(lambdas) @ U.T

b = np.random.randn(d)
w = np.linalg.solve(A,b)
# w = A\b # groundtruth

w0 = np.random.randn(d) # fix initializer
algs = ["Richardson", "Cheby", "Polyak", "ConjGrad", "Alg5"]
W = np.zeros((d, len(algs))) # can compare to the groundtruth w
pl = plt.figure()
plt.title("dim = {d}, $\kappa$ = {k}".format(d=d, k=κ))

#
wts = [None] * len(algs)
gts = [None ]* len(algs)
corrts = [None ]* len(algs)
acorrts = [None ]* len(algs)

for ii in range(len(algs)):
    print(W[:,ii].shape)
    # print(cheby.Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter)[0].shape)
    W[:,ii], obj, wts[ii], gts[ii], corrts[ii], acorrts[ii] = cheby.Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter)
    plt.plot(range(len(obj)), obj, label=algs[ii], lw=2)
    plt.xlabel("iteration")
    plt.ylabel("$\|A\mathbf{w} - \mathbf{b}\|_2$")
    
plt.legend()
plt.show()
plt.close()


for ii in range(len(algs)):
    # print(cheby.Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter)[0].shape)
    res = np.zeros(len(wts[ii]))
    for i in range(len(res)):
        res[i] = (wts[ii][i]-w).T @ gts[ii][i]

    plt.plot(range(len(wts[ii])), res, label=algs[ii], lw=2)
    plt.xlabel("iteration")
    plt.ylabel("$(w_t-w_{*})^{T}g_t$")
    
plt.legend()
plt.show()
plt.close()

for ii in range(len(algs)):
    # print(cheby.Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter)[0].shape)

    plt.plot(range(len(corrts[ii])), corrts[ii], label=algs[ii], lw=2)
    plt.xlabel("iteration")
    plt.ylabel("$g_{t}^{T}g_{t-1}$")
    
plt.legend()
plt.show()
plt.close()

for ii in range(len(algs)):
    # print(cheby.Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter)[0].shape)

    plt.plot(range(len(acorrts[ii])), acorrts[ii], label=algs[ii], lw=2)
    plt.xlabel("iteration")
    plt.ylabel("$g_{t}^{T}Ag_{t-1}$")
    
plt.legend()
plt.show()
plt.close()





# savefig(pl,"Cheby-$d-$κ.png")


# savefig(pl,"Cheby-$d-$κ.png")
