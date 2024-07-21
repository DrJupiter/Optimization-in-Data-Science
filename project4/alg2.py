#%% 
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

from file_helper import to_onehot, check_board

def matrix_one_hot(M):
    n = M.shape[0]
    O = np.array([[to_onehot(c, n) for c in r] for r in M])
    return O

def init_Z(X, N):
    Z = X.copy()
    NN = np.arange(1,N+1)
    rows, cols = Z.shape
    for i in range(rows):
        missing = np.array([j for j in NN if j not in X[i]])
        _Zi = Z[i]
        _Zi[X[i] == 0] = missing
        Z[i] = _Zi
    return matrix_one_hot(Z)

# %%

def proj(X,W):
    Z = np.zeros_like(X)
    NZR = np.array([i for i in range(X.shape[0]) if X[i].sum() == 0])
    NZC = np.array([j for j in range(X.shape[1]) if X[:,j].sum() == 0])
    Wp = W[NZR, :].copy()
    Wp = Wp[:, NZC]
    row_ind, col_ind= linear_sum_assignment(-Wp)

    Zp = Z[NZR, :]
    Zp = Zp[:, NZC]

    
    Zp[row_ind , col_ind] = 1

    for i,j in zip(NZR[row_ind], NZC[col_ind]):
        Z[i,j] = 1
    

    Z = X.copy() + Z
    return Z

# %%

def alg2(N, X, proj, maxiter=1e3):
    maxiter = int(maxiter)
    n = int(np.sqrt(N))

    Z = init_Z(X, N)
    X = matrix_one_hot(X)
    W = np.zeros_like(X)
    for t in range(maxiter):

        for k in range(N):
            W[:,:,k] = proj(X[:,:,k], Z[:,:,k])
        
        Z = Z - W
        W = W - Z
        for k in range(1,N+1):
            r = int(((k-1) % n) + 1)
            c = int(np.floor((k-1)/n) + 1)
            x = X[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n,:]
            w = W[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n,:]
            shape = w.shape

            w = proj(x.reshape(N,N), w.reshape(N,N))
            W[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n,:] = w.reshape(shape)
        
        Z = Z + W

        # termination check
        if t % 10 == 0:

            board = Z.copy()
             
            for k in range(N):
                board[:,:,k] = proj(X[:,:,k], board[:,:,k])
            board = np.argmax(board, 2) + 1
            if check_board(board):
                return board

    for k in range(N):
        Z[:,:,k] = proj(X[:,:,k], Z[:,:,k])
    return np.argmax(Z,2) + 1

# %%

if __name__ == "__main__":

    from file_helper import load_sudokus, compare_sudokus
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='File', type=str, nargs=2,
    help='unsolved board ground truth')
    args = parser.parse_args()
    unsolved_boards = load_sudokus(args.files[0])
    groundtruth_boards = load_sudokus(args.files[1])
    my_solution = []

    my_algorithm = alg2
    for start_board in unsolved_boards:
        answer = my_algorithm(start_board.shape[0],start_board, proj)
        my_solution.append(answer)

    compare_sudokus(my_solution,groundtruth_boards)