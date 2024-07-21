# %%
import argparse
import numpy as np
from file_helper import check_board

def init_Z(X, N):
    Z = X.copy()
    NN = np.arange(1,N+1)
    rows, cols = Z.shape
    for i in range(rows):
        missing = np.array([j for j in NN if j not in X[i]])
        _Zi = Z[i]
        _Zi[X[i] == 0] = missing
        Z[i] = _Zi
    return Z
    


def alg1(N, X, proj, maxiter=1e3):

    n = int(np.sqrt(N))

    Z = init_Z(X, N)

    maxiter = int(maxiter)

    # This should make it paralizable, however due to numpy's memory system
    # I was unable to create a proper copy of X that was not overwritten each time
    # So I do the computations sequentially
    # vproj = np.vectorize(proj, signature='(n),(n),()->(n)')
    for t in range(maxiter):
        # project for rows
        for i in range(N):
            Z[i] = proj(X[i], Z[i], N)

        for j in range(N):
            Z[:,j] = proj(X[:,j], Z[:,j], N)
        #Z = vproj(X, Z) 

        # project for col
        #Z.T = vproj(X.T, Z.T)

        for k in range(1, N+1):
            r = int(((k-1) % n) + 1)
            c = int(np.floor((k-1)/n) + 1)

            # subtract -1 for 0 index
            x = X[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n]
            z = Z[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n]
            shape = x.shape
            z = proj(x.flatten(),z.flatten(), N).reshape(shape)
            Z[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n] = z

        if t % 10 == 0:
            board = Z 
            if check_board(board):
                return board
    return Z

# %%

import copy
def proj(x, w, N):
    z = np.zeros_like(x)
    _x = copy.deepcopy(x)

    nz = [i for i in range(len(x)) if x[i] != 0]
    yz = [i for i in range(len(x)) if x[i] == 0]

    z[nz] = _x[nz]

    NN = np.arange(0,N) 

    o = sorted(NN[yz], key=lambda x: w[x], reverse=True)

    R = [n for n in range(1, N+1) if n not in x[nz]]
    z[o] = sorted(R, reverse=True)
    return z


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

    my_algorithm = alg1
    for start_board in unsolved_boards:
        answer = my_algorithm(start_board.shape[0],start_board, proj)
        my_solution.append(answer)

    compare_sudokus(my_solution,groundtruth_boards)





