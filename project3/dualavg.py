#%%
import numpy as np
import matplotlib.pyplot as plt


def func(X, Y, w):

    X = np.r_[ X, [np.ones(X.shape[1])] ]
    X = X.T

    n = len(X)

    inner = 1-X@w*y

    outer = np.amax((inner,np.zeros_like(inner)), axis=0)
    f = 1/n * np.sum(outer)

    inner_grad = -X*y.reshape(-1,1)

    inner_grad[inner < 0] = 0
    inner_grad[inner == 0] *= np.random.uniform(0,1)
    grad = 1/n * np.sum(inner_grad, axis=0)
    
    return f, grad


def proj(c, w):
    if np.linalg.norm(w) <= c:
        return w
    else:
        return w/np.linalg.norm(w) * c


from math import sqrt

def dual_avg(func, proj, X,y,w,s,c, n0=1, maxiter = int(1e3)):
    
    maxiter = int(maxiter)
    for t in range(maxiter):
        f, g = func(X, y, w)
        s = s + g
        eta = n0/sqrt(t+1) 
        w = proj(c,-eta*s)
    return w


#%% Initialize data

X = np.array([
    [1,0,-1,-1,-1.5],
    [1,-1,1,-1,0.5]
]) 
y = np.array([
    1,1,1,-1,-1
])

C = np.array([0.1,1,10])

w = np.zeros(X.shape[0]+1)
s = np.zeros(X.shape[0]+1)

W = []
for c in C:
    W.append(dual_avg(func, proj, X,y,w,s,c, n0=1, maxiter=1000))

#%%

w_stars = np.array(W)

for i, w_star in enumerate(w_stars):
    a = w_star[0]/w_star[1]
    b = w_star[-1]

    x_cords = np.linspace(-1.6,1.6,100)
    y_cords = x_cords*a+b

    point_colors = ["red" if yi == -1 else "blue" for yi in y]

    plt.plot(x_cords,y_cords,c="green")
    plt.scatter(X[0],X[1],c = point_colors)
    plt.title(f"Dual Average SVM, red=-1, blue=1, c={C[i]}")
    plt.show()