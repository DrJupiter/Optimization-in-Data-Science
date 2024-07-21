# %% dataset

from re import I
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
x1 = [1,1,-1,-1]
x2 = [1,-1,1,-1]
X = np.array([x1,x2]).T
y = np.array([1,1,1,-1])
N = X.shape[0]
d = X.shape[1]

# %% algorithm

def condgrad(func, pol, stepsize="const", tol=1e-3, maxiter=10**2):

    # Initialization for w0 in C
    wt = np.zeros(N)
    J = y == -1
    I = y == 1
    _ = wt[J]
    _[np.random.randint(0, sum(J))] = 1
    wt[J]+=_ 
    _ = wt[I]
    _[np.random.randint(0, sum(I))] = 1
    wt[I] += _
    assert np.sum(wt) == 2 and np.dot(wt, y) == 0 



    for t in range(int(maxiter)):
        ft, g = func(wt)
        #print(ft)
        zt = pol(g)
        if np.dot(wt - zt, g) <= tol:
            break
        if stepsize == "const":
            eta = 2/(t+2)
        else:
            eta = - np.dot(g,zt-wt)/np.dot(func_grad(zt-wt), zt-wt)
            #print(eta)
            eta = proj(eta, 0, 1)
        wt = (1-eta) * wt + eta * zt 
    
    return wt

def proj(x, low=0, high=1):
    # both of these should be the same, but 
    # we will use the latter given in the notes
    # just to be safe initially
    #return max(min(x, high),low)
    return min(high, max(x, low))

def func_grad(w):
    grad = np.array([ y[k] * sum([X[k,j] * sum([ w[i]*y[i]*X[i,j] for i in range(N)]) for j in range(d)]) for k in range(N) ])
    return grad

def func(w):
    sum_w = np.sum(np.array([ X[i] * w[i]*y[i] for i in range(N)]), axis=0)
    res = 1/2 * np.dot(sum_w, sum_w)
    assert_almost_equal(res, 1/2 * sum([ sum([w[i]*y[i]*X[i,j] for i in range(N)])**2 for j in range(d)]))
    #assert res == 1/2 * sum([ sum([w[i]*y[i]*X[i,j] for i in range(N)])**2 for j in range(d)])
    return res, func_grad(w)

def pol(grad):
    z = np.zeros(N)
    J = y == 1
    I = y == -1 
    _n = np.arange(0,N)

    j_s = (_n[J])[np.argmin(grad[J])]
    i_s = (_n[I])[np.argmin(grad[I])]
    z[[j_s, i_s]] = 1
    return z

# %% alpha*
a_s = condgrad(func, pol, stepsize='cauchy', maxiter=1e4, tol=0)
# %% plots and reconstruction

w_rec = sum([a_s[i] * y[i] * X[i] for i in range(N)])
_ = np.random.random()
b = np.max(1- X[y==1] @ w_rec) * (1-_) + _ * np.min(-1-X[y==-1] @ w_rec)
# %%
stretch=0.2
xx = np.linspace(min(X.T[0])-stretch, max(X.T[0])+stretch, 50).reshape(-1,1)
coords = w_rec * xx
x_c,y_c = coords.T[0], coords.T[1]
plt.plot(x_c,y_c, color="black", label="hyperplane")
pos = X[y==1]
neg = X[y==-1]
plt.scatter(pos.T[0], pos.T[1], color="blue", label="positve")
plt.scatter(neg.T[0], neg.T[1], color="red", label="negative")
plt.legend()
plt.title("Dual SVM $\eta$ derived from Cauchy's rule")
plt.show()



# %% Sanity check

from sklearn import svm

_pred = svm.LinearSVC(C=1).fit(X,y)

xx = np.linspace(min(X.T[0])-2, max(X.T[0])+2, 50).reshape(-1,1)
coords = _pred.coef_[0]* xx
x_c,y_c = coords.T[0], coords.T[1]
plt.plot(x_c,y_c, color="black", label="hyperplane")
pos = X[y==1]
neg = X[y==-1]
plt.scatter(pos.T[0], pos.T[1], color="blue", label="positve")
plt.scatter(neg.T[0], neg.T[1], color="red", label="negative")
plt.legend()
plt.show()
# %% Sequentual update
def condgrad_seq(func, pol, stepsize="const", tol=1e-3, maxiter=10**2):

    # Initialization for w0 in C
    wt = np.zeros(N)
    J = y == -1
    I = y == 1
    _ = wt[J]
    _[np.random.randint(0, sum(J))] = 1
    wt[J]+=_ 
    _ = wt[I]
    _[np.random.randint(0, sum(I))] = 1
    wt[I] += _
    assert np.sum(wt) == 2 and np.dot(wt, y) == 0 



    for t in range(int(maxiter)):

        ft, g = func(wt)
        #print(ft)
        zt = pol(g)
        if np.dot(wt - zt, g) <= tol:
            break
        if stepsize == "const":
            eta = 2/(t+2)
        else:
            eta = - np.dot(g,zt-wt)/np.dot(func_grad(zt-wt), zt-wt)
            eta = proj(eta, 0, 1)
        wt[J] = (1-eta) * wt[J] + eta * zt[J] 
 
        ft, g = func(wt)
        #print(ft)
        zt = pol(g)   
        
        if stepsize != "const":
            eta = - np.dot(g,zt-wt)/np.dot(func_grad(zt-wt), zt-wt)
            eta = proj(eta, 0, 1)

        wt[I] = (1-eta) * wt[I] + eta * zt[I] 

    return wt

# %% alpha*
a_s = condgrad_seq(func, pol, stepsize='const', maxiter=1e3, tol=1e-3)
# %% plots and reconstruction

w_rec = sum([a_s[i] * y[i] * X[i] for i in range(N)])
_ = np.random.random()
b = np.max(1- X[y==1] @ w_rec) * (1-_) + _ * np.min(-1-X[y==-1] @ w_rec)
# %%
stretch=0.2
xx = np.linspace(min(X.T[0])-stretch, max(X.T[0])+stretch, 50).reshape(-1,1)
coords = w_rec * xx
x_c,y_c = coords.T[0], coords.T[1]
plt.plot(x_c,y_c, color="black", label="hyperplane")
pos = X[y==1]
neg = X[y==-1]
plt.scatter(pos.T[0], pos.T[1], color="blue", label="positve")
plt.scatter(neg.T[0], neg.T[1], color="red", label="negative")
plt.legend()
plt.title("Sequential update with const update for $\eta$")
plt.show()
# %%
