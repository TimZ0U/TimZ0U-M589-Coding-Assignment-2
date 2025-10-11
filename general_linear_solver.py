import numpy as np

def paqlu_decomposition(A):

    A = np.array(A, copy=True)  
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")
    m, n = A.shape
    if not (np.issubdtype(A.dtype, np.floating) or np.issubdtype(A.dtype, np.complexfloating)):
        raise TypeError("A's dtype is not a supported numeric type (float or complex)")
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)
    r = 0  
    eps = np.finfo(A.dtype).eps if np.issubdtype(A.dtype, np.floating) else np.finfo(np.float64).eps
    for k in range(min(m, n)):
        sub_rows = P[k:m]
        sub_cols = Q[k:n]
        if sub_rows.size == 0 or sub_cols.size == 0:
            break  
        submatrix = A[np.ix_(sub_rows, sub_cols)]
        max_idx = np.unravel_index(np.nanargmax(np.abs(submatrix)), submatrix.shape)
        pivot_val = submatrix[max_idx]
        max_abs = np.abs(pivot_val)
        tol = max(m, n) * eps * np.abs(submatrix).max()  
        if max_abs <= tol:
            break
        pivot_row = sub_rows[max_idx[0]]
        pivot_col = sub_cols[max_idx[1]]
        if pivot_row != P[k]:
            piv_idx = np.where(P == pivot_row)[0][0] 
            P[k], P[piv_idx] = P[piv_idx], P[k]
        if pivot_col != Q[k]:
            piv_jdx = np.where(Q == pivot_col)[0][0]  
            Q[k], Q[piv_jdx] = Q[piv_jdx], Q[k]
        pivot = A[P[k], Q[k]]
        for i_idx in range(k+1, m):
            i = P[i_idx]            
            factor = A[i, Q[k]] / pivot  
            A[i, Q[k]] = factor     
            A[i, Q[k+1:n]] -= factor * A[P[k], Q[k+1:n]]
        r += 1  

    L = np.zeros((m, r), dtype=A.dtype)
    U = np.zeros((r, n), dtype=A.dtype)
    for i in range(m):
        for j in range(min(r, m)):  
            if i == j and j < r:
                L[i, j] = 1  
            elif i > j and j < r:
                L[i, j] = A[P[i], Q[j]] 
    for i in range(r):
        for j in range(n):
            if i <= j:
                U[i, j] = A[P[i], Q[j]]  
            else:
                U[i, j] = 0  
    return P.copy(), Q.copy(), L, U

def paqlu_decomposition_in_place(A):

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")
    m, n = A.shape
    if not (np.issubdtype(A.dtype, np.floating) or np.issubdtype(A.dtype, np.complexfloating)):
        raise TypeError("A's dtype is not a supported numeric type (float or complex)")
    P, Q, L, U = paqlu_decomposition(A)
    r = L.shape[1]  
    A[...] = 0
    for i in range(r):
        row = P[i]
        for j in range(n):
            col = Q[j]
            if i <= j < n:
                A[row, col] = U[i, j]
    for j in range(r):
        col = Q[j]
        for i_idx in range(j+1, m):
            A[P[i_idx], col] = L[P[i_idx], j] if j < L.shape[1] else 0
    return P, Q  

def solve(A, b):
    """
    Solve A x = b for x = N @ x_free + c, where columns of N form a basis of nullspace and c is a particular solution.
    Returns (N, c).
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    elif b.ndim > 2:
        raise ValueError("b must be a 1D or 2D array (vector or matrix of RHS)")
    if b.shape[0] != m:
        raise ValueError("Incompatible dimensions: A is %d×%d, but b is length %d" % (m, n, b.shape[0]))
    if m == 0:
        if b.size != 0:
            raise ValueError("Inconsistent system: A has 0 rows but b is not empty")
        N = np.eye(n, dtype=A.dtype)
        c = np.zeros((n, b.shape[1]), dtype=A.dtype) if b.shape[1] > 1 else np.zeros(n, dtype=A.dtype)
        return N, c
    if n == 0:
        if np.any(b != 0):
            raise ValueError("inconsistent system: A x = b has no solution (A has 0 columns but b is nonzero)")
        N = np.empty((0, 0), dtype=A.dtype)
        c = np.empty((0, b.shape[1]), dtype=A.dtype) if b.shape[1] > 1 else np.empty((0,), dtype=A.dtype)
        return N, c
    P, Q = paqlu_decomposition_in_place(A)
    r = 0
    min_mn = min(m, n)
    for i in range(min_mn):
        if abs(A[P[i], Q[i]]) <= max(m, n) * np.finfo(A.dtype).eps * np.abs(A[P[i:], :][:, Q[i:]]).max():
            break
        r += 1
    b_permuted = b[P, :]
    y = np.array(b_permuted, dtype=A.dtype, copy=True)  
    for i in range(m):
        j_max = min(i, r)
        for j in range(j_max):
            y[i, :] -= A[P[i], Q[j]] * y[j, :]
        if i < r:
            pass
        else:
            if np.linalg.norm(y[i, :], ord=np.inf) > max(m, n) * np.finfo(A.dtype).eps * np.linalg.norm(b_permuted, ord=np.inf):
                raise ValueError("inconsistent system: A x = b has no solution")
    z = np.zeros((r, b.shape[1]), dtype=A.dtype)
    for i in range(r-1, -1, -1):
        z[i, :] = y[i, :]
        for j in range(i+1, r):
            z[i, :] -= A[P[i], Q[j]] * z[j, :]
        pivot_val = A[P[i], Q[i]]
        if abs(pivot_val) <= np.finfo(A.dtype).eps * 10:  # check for near-zero pivot
            raise ValueError("inconsistent system: singular pivot encountered")
        z[i, :] /= pivot_val
    x_perm = np.vstack([z, np.zeros((n - r, b.shape[1]), dtype=A.dtype)])

    Q_inv = np.argsort(Q)
    c_full = x_perm[Q_inv, :]
    if c_full.shape[1] == 1:
        c_full = c_full[:, 0]
    f = n - r  
    if f <= 0:
        N = np.empty((n, 0), dtype=A.dtype)
    else:
        N = np.zeros((n, f), dtype=A.dtype)
        free_indices_perm = np.arange(r, n)  
        for idx, j in enumerate(free_indices_perm):

            y_free = -A[P[0:r], Q[j]]
            h = np.zeros(r, dtype=A.dtype)
            for i in range(r-1, -1, -1):
                temp = y_free[i]
                for col in range(i+1, r):
                    temp -= A[P[i], Q[col]] * h[col]
                h[i] = temp / A[P[i], Q[i]]
            x_free = np.zeros(n, dtype=A.dtype)
            x_free[0:r] = h
            x_free[j] = 1.0
            N[:, idx] = x_free[Q_inv]
    return N, c_full
