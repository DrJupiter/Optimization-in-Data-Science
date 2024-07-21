# %%
import random
import numpy as np
import pickle
with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

# NxD
train_x = train_data[:,:23]
factors = 1/np.amax(train_x,0)
train_x = train_x*factors

# Nx1
train_y = train_data[:,23] # labels are either 0 or 1
test_x = test_data[:,:23]
test_y = test_data[:,23] # labels are either 0 or 1
# %%


# log f

def sgda(X,y, proj_w,proj_a, l, w0, a0, eta_t, maxiter=1e3):

    n = len(y)
    a = a0
    w = w0

    maxiter = int(maxiter)

    f_vals = np.zeros(maxiter//100)
    f_avg = np.zeros(maxiter//100)
    f_u_bar = np.zeros(maxiter//100)
    f_l_bar = np.zeros(maxiter//100)

    w_avg = 0
    a_avg = 0

    t_w_avg = 0
    t_a_avg = 0


    eta_sum = 0

    for t in range(1, maxiter+1):



            

        i = random.randint(0,n-1) 
        eta = eta_t(t)

        # for avg
        eta_sum += eta
        t_w_avg += w*eta
        t_a_avg += a*eta

        w_avg = t_w_avg/eta_sum
        a_avg = t_a_avg/eta_sum


        if (t-1) % 100 == 0:
            idx = (t-1)//100
            f_vals[idx] = f(X,y,l,a,w)
            f_avg[idx] = f(X,y,l,a_avg,w_avg)
            f_u_bar[idx] = f_lower(w, l, X, y)- f_upper(a, l, X, y)
            f_l_bar[idx] = f_lower(w_avg, l, X, y)- f_upper(a_avg, l, X, y)

        
        w_n = proj_w(w - eta * fw_grad(l, w,a[i], X[i], y[i]))

        a_n = a
        a_n[i] = proj_a(a[i]  + eta * fa_grad(w,a[i], X[i], y[i]))

        a = a_n
        w = w_n
    return w, a, f_vals, f_avg, f_u_bar, f_l_bar


def fw_grad(l, w,ai, x, y):
    return l*w-ai * x * y 

def fa_grad(w, ai, x, y):
    # potential crashes for ai=0 or 1, maybe add a small epislon here to combat this
    eps = np.finfo(float).eps
    return - (np.dot(y*x, w) + np.log(ai + eps) - np.log(1-ai + eps))

def proj_w(w):
    return w

def proj_a(a):
    return min(1,max(0,a))

def eta_t_one(t):
    return 1/t

def eta_t_two(t):
    return 1/np.sqrt(t)


# %%

def f_term(x,y,w,a):
    one = - a * np.dot(y*x,w)
    two = 0 if a == 0 else -a*np.log(a)
    three = 0 if a == 1 else -(1-a)*np.log(1-a)
    return one+two+three

def f(X, y, l, a, w):
    n = len(y)
    return l/2 * np.dot(w, w) + 1/n * np.sum([f_term(X[i], y[i], w, a[i]) for i in range(n)])

def f_upper(a, l, X, y):
    n = len(y)
    term = 1/(n*l) * np.sum([a[i]*y[i]*X[i] for i in range(n)]) 
    return f(X, y, l, a, term)

def f_lower(w, l, X, y):
    n = len(y)
    return l/2 * np.dot(w,w) + 1/n * np.sum([np.log(1+np.exp(-y[i]*np.dot(X[i],w))) for i in range(n)]) 

# %%

#w_s, a_s, f_vals, f_avg, f_u_bar, f_l_bar  = sgda(train_x,train_y, proj_w, proj_a, 1, np.ones_like(train_x[0]), np.ones(len(train_y))-0.2, eta_t_one, 500)

# %%


# %%
w_s, a_s, f_vals, f_avg, f_u_bar, f_l_bar  = sgda(train_x,train_y, proj_w, proj_a, 1, np.ones_like(train_x[0]), np.ones(len(train_y))-0.2, eta_t_one, 6*20e3)

import matplotlib.pyplot as plt
interval = range(len(f_vals))
plt.plot(interval, f_vals, label = "$f(w_t,\\alpha_t)$")
plt.plot(interval, f_avg, label = "$f(\\bar{w}_t,\\bar{\\alpha}_t)$")
plt.plot(interval, f_u_bar, label = "$f^{\_}(w) - f_{\_}(\\alpha)$")
plt.plot(interval, f_u_bar, label = "$f^{\_}(\\bar{w}) - f_{\_}(\\bar{\\alpha})$")
plt.legend()
plt.xlabel("x/1000")
plt.title("$\\eta = \\frac{1}{t}$")

# %%

w_s, a_s, f_vals, f_avg, f_u_bar, f_l_bar  = sgda(train_x,train_y, proj_w, proj_a, 1, np.ones_like(train_x[0]), np.ones(len(train_y))-0.2, eta_t_two, 20e3)

import matplotlib.pyplot as plt
interval = range(len(f_vals))
plt.plot(interval, f_vals, label = "$f(w_t,\\alpha_t)$")
plt.plot(interval, f_avg, label = "$f(\\bar{w}_t,\\bar{\\alpha}_t)$")
plt.plot(interval, f_u_bar, label = "$f^{\_}(w) - f_{\_}(\\alpha)$")
plt.plot(interval, f_u_bar, label = "$f^{\_}(\\bar{w}) - f_{\_}(\\bar{\\alpha})$")
plt.legend()
plt.xlabel("x/1000")
plt.title("$\\eta = \\frac{1}{\\sqrt{t}}$")


# %%
import matplotlib.pyplot as plt
interval = range(len(f_vals)-1)
plt.plot(interval, f_vals[1:], label = "$f(w_t,\\alpha_t)$")
plt.plot(interval, f_avg[1:], label = "$f(\\bar{w}_t,\\bar{\\alpha}_t)$")
plt.plot(interval, f_u_bar[1:], label = "$f^{\_}(w) - f_{\_}(\\alpha)$")
plt.plot(interval, f_u_bar[1:], label = "$f^{\_}(\\bar{w}) - f_{\_}(\\bar{\\alpha})$")
plt.legend()
plt.xlabel("x/1000")
plt.title("$\\eta = \\frac{1}{\\sqrt{t}}$")


# %%
