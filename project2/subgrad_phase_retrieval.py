import random
from data_generation import datagen
import numpy as np
 
np.random.seed(42)
random.seed(42)

def subgradient(func, w, X, y, fb: str, tol=1e-3, maxiter=10**3):
    f_star = 0
    maxiter = int(maxiter)
    wt_log = np.zeros((maxiter, w.shape[0]))
    f_log = np.zeros((maxiter))
    for t in range(maxiter):

        ft, g = func(w, X, y)

        wt_log[t] = w
        f_log[t] = ft

        g_euclid = np.linalg.norm(g)
        if g_euclid <= tol:
            print("Converged")
            return wt_log[0:t], f_log[0:t]

        w = w - (ft-f_star)/g_euclid * g/g_euclid

        if fb == "ub":
            if t == 0:
                f_star = ft + tol
            f_star = min(f_star, ft) + tol
    return wt_log, f_log

def func(w, X, y):
    # sum
    N = X.shape[0]
    res = 1/N * sum([np.abs(np.dot(X[i], w)**2 - y[i]) for i in range(N)])
    sub_grad = func_grad(w, X, y)
    return res, sub_grad

def func_grad(w, X, y):
    N = X.shape[0]
    res = 1/N * sum([l_subdiff(w, X[i], y[i]) for i in range(N)])
    return res

def l_subdiff(w, x, y):
    inner = np.dot(x,w)**2 - y
    
    diff = np.dot(x,w) * 2 * x
    if inner > 0:
        return diff
    elif inner < 0:
        return - diff
    else:
        # random number in interval [-1;1] 
        #t = (1- (-1)) * random.uniform(0,1) + (-1)
        t = random.uniform(-1, 1)
        return diff * t

        


if __name__ == "__main__":
    #  
    # two cases of f* 
    import matplotlib.pyplot as plt
    n_l = [20,100,200,500,1000]
    
    for n in n_l:
    #for n in [n_l[0]]:

        X, y, w_star = datagen(n,d=10,p=0)
        w0 = np.random.standard_normal(size=w_star.shape[0]) 
        w_log, f_log = subgradient(func, w0, X, y, "")
        norm_w_star = np.linalg.norm(w_star)
        log_error = np.array([np.log(np.linalg.norm(wt-w_star)/norm_w_star) for wt in w_log])

        #log_error = np.array([np.linalg.norm(wt-w_star) for wt in w_log])

        plt.plot(range(len(log_error)), log_error, label=f"{n}")
        #plt.plot(range(len(log_error)), np.exp(log_error), label=f"{n}")
        #plt.plot(range(len(log_error)), f_log, label=f"{n}")

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Log Error $\log(||w_t-w_{*}||_2/||w_{*}||_2)$")
    plt.title("Subgradient with $p=0, f_{*}=0, d=10, n \in [20,100,200,500,10^3], w \sim \mathcal{N}_d(0, I)$")
    plt.show()
    plt.close()

    for n in n_l:
    #for n in [n_l[0]]:

        X, y, w_star = datagen(n,d=10,p=0.1, sigma=10)
        w0 = np.random.standard_normal(size=w_star.shape[0]) 
        w_log, f_log = subgradient(func, w0, X, y, "")
        norm_w_star = np.linalg.norm(w_star)
        log_error = np.array([np.log(np.linalg.norm(wt-w_star)/norm_w_star) for wt in w_log])

        #log_error = np.array([np.linalg.norm(wt-w_star) for wt in w_log])

        plt.plot(range(len(log_error)), log_error, label=f"{n}")
        #plt.plot(range(len(log_error)), np.exp(log_error), label=f"{n}")
        #plt.plot(range(len(log_error)), f_log, label=f"{n}")

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Log Error $\log(||w_t-w_{*}||_2/||w_{*}||_2)$")
    plt.title("Subgradient with $p=0.1, f_{*}=0, d=10, n \in [20,100,200,500,10^3], w \sim \mathcal{N}_d(0, I), \sigma=10$")
    plt.show()
    plt.close()
 
    for n in n_l:
    #for n in [n_l[0]]:

        X, y, w_star = datagen(n,d=10,p=0.1, sigma=10)
        w0 = np.random.standard_normal(size=w_star.shape[0]) 
        w_log, f_log = subgradient(func, w0, X, y, "ub")
        norm_w_star = np.linalg.norm(w_star)
        log_error = np.array([np.log(np.linalg.norm(wt-w_star)/norm_w_star) for wt in w_log])

        #log_error = np.array([np.linalg.norm(wt-w_star) for wt in w_log])

        plt.plot(range(len(log_error)), log_error, label=f"{n}")
        #plt.plot(range(len(log_error)), np.exp(log_error), label=f"{n}")
        #plt.plot(range(len(log_error)), f_log, label=f"{n}")

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Log Error $\log(||w_t-w_{*}||_2/||w_{*}||_2)$")
    plt.title("Subgradient with $p=0.1, f_{*}=\min(f_t \text{ so far}), d=10, n \in [20,100,200,500,10^3], w \sim \mathcal{N}_d(0, I), \sigma=10$")
    plt.show()   
    plt.close()