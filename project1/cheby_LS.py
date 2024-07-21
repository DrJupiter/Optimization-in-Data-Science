import numpy as np

def Cheby_LS(A, b, σ, L, alg="Cheby", w=None, maxiter=10**3, tol=1e-5):
    if w is None:
        w = np.random.randn(len(b))
# The Chebyshev algorithm for solving linear system Aw = b
    # A: square matrix, positive definite
    # b: vector
    # σ: lower bound of the spectrum of A
    # L: upper bound of the spectrum of A
    # w: initial guess
    # alg: "Richardson", "Cheby" (default), "Polyak", "ConjGrad"
    # maxiter: maximum number of iteration; default 10^3
    # tol: tolerance; default 1e-5

    z = w.copy() # backup the previous iterate

    g = A @ z - b # gradient # g0

    ϵ = np.finfo(float).eps # smallest number to prevent dividing by 0
    κ = L / (σ + ϵ) # condition number
    c = ( (κ-1) / (κ+1) / 2 )**2
    η = 2. / (L+σ+ϵ) # step size
    γ = 2. # momentum size
    if alg == "Richardson":
        c = 0
    elif alg == "Polyak":
        γ = 2*(κ+1) / (np.sqrt(κ)+1)**2
    elif alg == "ConjGrad":
        γ = 1. # don't forget this! # y0
        print("A shape: " + str(A.shape) + " | g shape: " + str(g.shape))
        η = np.dot(g,g) / (np.dot(A@g,g)+ϵ) # n0
        ζ = η # backup step size # will function as n(t-1)
    elif alg == "Alg5":
        γ = 1. # don't forget this! # y0
        η = (np.dot(g.T, A@g))/(np.dot(A@g,A@g)+ϵ) # <g_t,g_t>/<Ag,g>
        ζ = η # backup step size # will function as n(t-1)
        # 

    w = z - η*g # Richardson step # w1
    # consider initializing w different for alg5
    print("w shape: "+ str(w.shape))

    if alg == "alg5":
        # redundant step, but part of my debugging.
        w = z - η * γ * g + (γ - 1) * (z - 0)

    obj = np.zeros(maxiter) # let's roll
    a_length = np.zeros(maxiter)

    # for point 2

    w_tracker = np.zeros((maxiter,z.shape[0]))
    g_tracker = np.zeros((maxiter,g.shape[0]))

    # correlation
    corr_tracker = np.zeros(maxiter)

    # gtAgt-1
    acorr_tracer = np.zeros(maxiter)

    for t in range(maxiter):
        obj[t] = np.linalg.norm(g) # evaluated at z (g_t-1), t starts at 1, not w

        # alg5
        a_length[t] = np.dot(g.T, A@g) # ||gt-1||^2_A

        # point 2 
        w_tracker[t] = z
        g_tracker[t] = g

        g = A @ w - b # g_t

        # record correlations
        corr_tracker[t] = np.dot(g.T, g_tracker[t])
        acorr_tracer[t] = np.dot(g.T, A@g_tracker[t])

        # mission accomplished
        if obj[t] <= tol:
            return z, obj[0:t], w_tracker[0:t], g_tracker[0:t], corr_tracker[0:t], acorr_tracer[0:t]


        if alg == "ConjGrad":
            η = np.dot(g,g) / (np.dot(A@g,g)+ϵ) # <g_t,g_t>/<Ag,g>
            γ = 1. / ( 1 - η*np.dot(g,g) / (ζ*obj[t]**2*γ+ϵ) ) # n_t-1 * ||g_t-1||^2 * y_t-1/ (1-n_t*<g_t,g_t>)
            ζ = η # update n_t-1 = next n_t-1
        elif alg == "Alg5":
            η = (np.dot(g.T, A@g))/(np.dot(A@g,A@g)+ϵ) # <g_t,g_t>/<Ag,g>
            
            γ = 1/(1 - (η*np.dot(A@g,g) / (ζ*a_length[t]*γ+ϵ))) # n_t-1 * ||g_t-1||^2 * y_t-1/ (1-n_t*<g_t,g_t>)
            #obj[t] = gt-1
            #ζ = n-t-1
            ζ = η # update n_t-1 = next n_t-1

            None
        else:
            γ = 1. / (1 - c*γ + ϵ)


        # mission halted
        if t == maxiter-1:
            return z, obj, w_tracker, g_tracker, corr_tracker, acorr_tracer


        # momentum update and backup w
        w, z = w - γ*η*g + (γ-1)*(w - z), w 



