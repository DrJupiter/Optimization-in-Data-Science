# %% 
import numpy as np
import matplotlib.pyplot as plt

# %%
import pickle
with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

train_x = train_data[:,:23]
factors = 1/np.amax(train_x,0)
train_x = train_x*factors
train_y = train_data[:,23] # labels are either 0 or 1
train_y = train_y * 2 - 1

test_x = test_data[:,:23]
test_x = test_x * factors
test_y = test_data[:,23] # labels are either 0 or 1
test_y = test_y * 2 - 1
# %%

n = len(train_x)
d = train_x.shape[1] - 1

#partial_L_sum = np.sqrt(2)/n * np.sum([(sum([ sum([ train_x[i,j] * train_x[i,k] for i in range(n)]) for k in range(d+1)]))**2 for j in range(d+1)])
partial_L_sum = np.sqrt(2)/n #* np.sum([(sum([ train_x[i,j] for i in range(n)]))**2 for j in range(d+1)])

# %%

# %%

def fastgd(func, w0, type, lamb, X, y, theta0=1, maxiter=1e2):
    z0 = w0
    maxiter = int(maxiter)

    log_ft = np.zeros(maxiter)
    log_gt = np.zeros(maxiter)
    log_wt = np.zeros((maxiter,len(w0)))

    w_t = w0
    z_t = z0
    theta_t = theta0
    for t in range(maxiter):

        ft, gt = func(w_t, lamb, X, y)

        log_ft[t] = ft
        log_gt[t] = np.linalg.norm(gt)
        log_wt[t] = w_t

        if t == 0:
            X_tilde = np.r_[X.T,[np.ones(len(X))]].T
            II = [t_plus(np.sign(1-yi*y_hat(xi, w_t))) for xi, yi in zip(X_tilde,y)]
            XX = np.array([[np.sum([X_tilde[i,r] * X_tilde[i, k] * II[i] for i in range(n)]) for k in range(X_tilde.shape[1])] for r in range(X_tilde.shape[1])]) 

            L = 4 * lamb + np.sqrt(np.max(np.linalg.eigvals(2/len(train_x) * XX))) + 10 
            print(L)
            eta = 1/L
        eta_t = eta
        z_p = z_t
        z_t = w_t-eta_t * gt 

        if type == "gd":
            w_t = z_t
        elif type == "fast":
            theta_t_n = (1+np.sqrt(4*theta_t**2 + 1))/2
            w_t = z_t + (theta_t-1)/(theta_t_n) * (z_t-z_p) 
            theta_t = theta_t_n
        elif type == "opt":

            if t < maxiter:
                theta_t_n = (1+np.sqrt(4*theta_t**2+1))/2
            else:
                theta_t_n = (1+np.sqrt(8*theta_t**2+1))/2
            w_t = z_t + (theta_t-1)/(theta_t_n)*(z_t-z_p) + theta_t/(theta_t_n) * (z_t-w_t)
            theta_t = theta_t_n
    return log_ft, log_gt, log_wt
# %%

def t_plus(v):
    return max((v,0))

def y_hat(x,w):
    return np.dot(x,w)

def func(w, lamb, X, y):

    n = len(X)

    X_tilde = np.r_[X.T,[np.ones(len(X))]].T

    f_t = 1/n * np.sum([t_plus(1-y_hat(xi,w)*yi)**2 for xi, yi in zip(X_tilde,y)]) + lamb * np.dot(w,w)
    w_ = w.copy()
    w_[-1] = 0
    g_t = 2 * lamb * w_ + 1/n * sum([-yi*xi * t_plus(2*(1-y_hat(xi,w))) for xi, yi in zip(X_tilde,y)]) 
    return f_t, g_t



# %%

LAMP = [0, 1, 10]


func_vals, grad_vals, W = [], [], []

types = ['gd', 'fast', 'opt']

for type in types:
    for lamb in LAMP:

        ff, gg, ww = fastgd(func, np.ones(train_x.shape[1]+1), type, lamb, train_x, train_y)
        func_vals.append(ff)
        grad_vals.append(gg)
        W.append(ww)

# %%

def evaluate(func, lamd, w_log, X, y):
    maxiter = int(100)

    log_ft = np.zeros(maxiter)
    log_gt = np.zeros(maxiter)

    for t in range(maxiter):
        w_t = w_log[t]
        ft, gt = func(w_t, lamb, X, y)

        log_ft[t] = ft
        log_gt[t] = np.linalg.norm(gt)
    
    return log_ft, log_gt


# %%
func_vals_test, grad_vals_test = [], []

kk = 0
for type in types:
    for lamb in LAMP:
        tff, tgg = evaluate(func, lamb, W[kk], test_x, test_y)
        func_vals_test.append(tff)
        grad_vals_test.append(tgg)
        kk += 1

# %%

nn = 0
for type in types:
    for lamp in LAMP:
        plt.plot(range(len(func_vals_test[nn])), func_vals_test[nn], label = f"{type},{lamp}")


        nn +=1
plt.title("Func values test")
plt.legend()
plt.show()
plt.close()





# %%
nn = 0
for type in types:
    for lamp in LAMP:
        plt.plot(range(len(grad_vals_test[nn])), grad_vals_test[nn], label = f"{type},{lamp}")


        nn +=1
plt.title("grad values test")
plt.legend()
plt.show()
plt.close()




# %%

nn = 0
for type in types:
    for lamp in LAMP:
        plt.plot(range(len(func_vals[nn])), func_vals[nn], label = f"{type},{lamp}")


        nn +=1
plt.title("Func values")
plt.legend()
plt.show()
plt.close()





# %%
nn = 0
for type in types:
    for lamp in LAMP:
        plt.plot(range(len(grad_vals[nn])), grad_vals[nn], label = f"{type},{lamp}")


        nn +=1
plt.title("grad values")
plt.legend()
plt.show()
plt.close()




# %%
