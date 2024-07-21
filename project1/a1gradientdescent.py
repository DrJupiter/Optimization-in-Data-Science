import numpy as np

def h_tau(s,tau):
    """
    The definition for the piecewise function h_{τ}
    """
    if np.abs(s) <= tau:
        return s**2/2
    elif np.abs(s) > tau:
        return tau*(np.abs(s)-tau/2)

def grad_h_tau(s,x,y,w,tau):
    """
    The gradient of h_{τ} as calculated in Part two Q1
    """
    if np.abs(s) <= tau:
        return (s) * x 
    elif np.abs(s) > tau:
        if s > 0:
            return tau*x 
        elif s < 0:
            return -tau*x 
        else:
            return np.random.uniform(low=-1, high=1) * tau * x


def gdhr(y,x, w, eta =1, ls=True, tol=1e-3, maxiter=1e3, tau = 1):
    """
    Gradient Descent for Huber Regression
    """
    maxiter = int(maxiter)

    # Extend all x with a one
    X = np.row_stack((x, np.ones((1,x.shape[1]))))

    # Define f as defined in the assignment
    def f(w, X, y):
        n = X.shape[1]
        return 1/n * np.sum([h_tau(np.dot(X[:,i].T, w) - y[i], tau) for i in range(n)])

    # The gradient of f as computed
    def grad_f(w, X, y):
        n = X.shape[1]
        _t = [grad_h_tau(np.dot(X[:,i].T, w), X[:,i], y[i], w, tau) for i in range(n)]
        return 1/n * np.sum([grad_h_tau(np.dot(X[:,i].T, w) - y[i], X[:,i], y[i], w, tau) for i in range(n)],axis =0)

    # Returns both f,  ∇f
    def func(w, X, y):
        return f(w, X, y), grad_f(w, X, y)        

    # List to store f_t
    obj = np.zeros(maxiter)
    w_tracker = np.zeros((maxiter,w.shape[0]))

    for t in range(maxiter):

        ft, g = func(w, X, y)

        # Record for trackers
        obj[t] = ft
        w_tracker[t] = w

        # Check for convergence
        if np.linalg.norm(g) <= tol:
            print("Converged")
            return obj[0:t+1], w_tracker[0:t]

        # w_t+1 = w - η_t * g
        _w = w - eta * g

        # Simply a status check on t        
        if t % 1e5 == 0:
            print(t)

        # Backtracking
        if ls:
            h, _ = func(_w, X, y) 

            # while not <=
            # f(w - η_t * g) <= ft - η_t * α_t  * ||g||_2^2
            while h > ft - eta * 1/2 * np.dot(g,g):
                eta = eta/2
                _w = w - eta * g
                h, _ = func(_w, X, y)

            # After a suitible step-size has been found
            # then update w_t+1 = w - η_t * g
            w = _w
        else:
            # If we aren't backtracking just go through with
            # the update
            w = _w

    return obj, w_tracker


def load_data(file_name):
    """
    Loads data from csv files.
    Transposes data for X
    """
    data = np.genfromtxt("./data/"+file_name, delimiter=',')
    if "X" in file_name:
        data = data.transpose()

    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load training data
    x,y = load_data("housing_X_train.csv"), load_data("housing_y_train.csv")

    # Constant step-size
    tau = 1
    L = np.linalg.norm(1/(x.T.shape[1]+1) * (tau+np.sum([x.T[:,i].T * tau for i in range(x.T.shape[1])], axis=0)))
    eta = np.random.uniform(low=0+0.1,high=2/L-0.1)/2 * 10**(-7)
    obj, w_tracker = (gdhr(y,x.T,np.zeros(x.T.shape[0]+1),eta=eta, maxiter=1e3, ls=False))

    # Load test data and define f for testing
    def f(w, X, y, tau=1):
        n = X.shape[1]
        return 1/n * np.sum([h_tau(np.dot(X[:,i].T, w) - y[i], tau) for i in range(n)])
    
    x_test,y_test = load_data("housing_X_test.csv"), load_data("housing_y_test.csv")
    X_test = np.row_stack((x_test.T, np.ones((1,x_test.T.shape[1]))))

    average_err = [f(w, X_test, y_test) for w in w_tracker]
    average_sqe_err = [f(w, X_test, y_test, tau=np.inf) for w in w_tracker]

    plt.plot(range(len(obj)),obj)
    plt.xlabel("iteration")
    plt.ylabel("$f(w_t)$")
    plt.show()
    plt.close()


    plt.plot(range(len(average_err)),average_err)
    plt.xlabel("iteration")
    plt.ylabel("Average $h_{\\tau}(\hat{y}-y)$ over test set")
    plt.show()
    plt.close()


    plt.plot(range(len(average_sqe_err)),average_sqe_err)
    plt.xlabel("iteration")
    plt.ylabel("Average $h_{\infty}(\hat{y}-y)$ over test set")
    plt.show()
    plt.close()

    obj, w_tracker = (gdhr(y,x.T,np.zeros(x.T.shape[0]+1), maxiter=1e3, ls=True))

    average_err = [f(w, X_test, y_test) for w in w_tracker]
    average_sqe_err = [f(w, X_test, y_test, tau=np.inf) for w in w_tracker]

    plt.plot(range(len(obj)),obj)
    plt.xlabel("iteration")
    plt.ylabel("$f(w_t)$")
    plt.show()
    plt.close()


    plt.plot(range(len(average_err)),average_err)
    plt.xlabel("iteration")
    plt.ylabel("Average $h_{\\tau}(\hat{y}-y)$ over test set")
    plt.show()
    plt.close()


    plt.plot(range(len(average_sqe_err)),average_sqe_err)
    plt.xlabel("iteration")
    plt.ylabel("Average $h_{\infty}(\hat{y}-y)$ over test set")
    plt.show()
    plt.close()


    
    