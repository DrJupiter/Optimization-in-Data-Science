# %%
import argparse
import numpy as np

from file_helper import init_from_board, convert_to_3d_repr, convert_to_board, check_board

def init_Z(X, N):
    Z = X.copy()
    NN = np.arange(1,N+1)
    rows, cols = Z.shape
    for i in range(rows):
        missing = np.array([j for j in NN if j not in X[i]])
        _Zi = Z[i]
        _Zi[X[i] == 0] = missing
        Z[i] = _Zi
    return convert_to_3d_repr(Z)

# %%
def p_inner(w):
    return w/np.dot(np.ones_like(w), w)

def proj(W, N, maxiter=1e2):
    maxiter=int(maxiter)

    for _ in range(maxiter):
        if _ % 2 == 0:
            for k in range(N):
                W[k] = p_inner(W[k])
        else:
            for k in range(N):
                W.T[k] = p_inner(W.T[k])
    return W




# %%

def alg5(N, X, proj, maxiter=1e3):

    maxiter = int(maxiter)
    n = int(np.sqrt(N))

    #Z = init_Z(X, N)
    X = init_from_board(X)
    Z = np.zeros_like(X)

    for i in range(N):
        for j in range(N):
            if X[i,j].sum() != 0:
                Z[i,j] = X[i,j].copy() 
            else:
                #Z[i,j] = np.ones_like(X[i,j]) * 1/N
                Z[i,j] = np.zeros_like(X[i,j])
    for t in range(maxiter):

        for i in range(N):
            Z[i,:,:] = proj(Z[i,:,:], N)
        for j in range(N):
            Z[:,j,:] = proj(Z[:,j,:], N)
        for k in range(1,N+1):
            r = int(((k-1) % n) + 1)
            c = int(np.floor((k-1)/n) + 1)

            z = Z[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n,:]
            shape = z.shape

            z = proj(z.reshape(N, N), N)

            Z[(r-1)*n+1-1:r*n, (c-1)*n+1-1:c*n,:] = z.reshape(shape)
        if t % 10 == 0:
            board = np.argmax(Z,2) + 1
            if check_board(board):
                return board
    return np.argmax(Z,2)+ 1 
    return convert_to_board(Z)
         
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

    my_algorithm = alg5
    for start_board in unsolved_boards:
        answer = my_algorithm(start_board.shape[0],start_board, proj)
        my_solution.append(answer)
    
    compare_sudokus(my_solution,groundtruth_boards)
