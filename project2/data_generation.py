import numpy as np
from numpy.testing import assert_almost_equal

def datagen(n=20, d=10, p=0, sigma=10):
    """
    The generation of the variables follows the specification given in the 
    assignment and uses distributions provied by numpy's random module.
    """
    X = np.array([np.random.standard_normal(size=d) for _ in range(n)])
    w_star = np.random.standard_normal(size=d) 
    z = np.random.binomial(n=1,p=p,size=n)
    eps = np.random.normal(loc= 0, scale =sigma, size=n) 
    y = (1-z)* (X @ w_star)**2 + z * np.abs(eps)

    # Test
    #_y = np.array([(1-z[i])* np.dot(X[i],w_star)**2 + z[i] * np.abs(eps[i]) for i in range(n)])
    #assert_almost_equal(sum(y-_y), 0)
    return X, y, w_star

if __name__ == "__main__":
    datagen()