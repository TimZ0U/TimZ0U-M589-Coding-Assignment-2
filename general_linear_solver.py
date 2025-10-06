import numpy as np

def paqlu_decomposition_in_place(A):
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    if not np.issubdtype(A.dtype, np.inexact):
        raise TypeError("A must have a floating or complex dtype")

    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)
    TOL = 1e-6
    r = 0

    for k in range(min(m, n)):

        col_entries = np.abs([A[P[i], Q[k]] for i in range(k, m)])
        pivot_rel_row = col_entries.argmax()
        pivot_row = k + pivot_rel_row
        pivot_val = A[P[pivot_row], Q[k]]

        if abs(pivot_val) <= TOL:
            break

        r += 1

        if pivot_row != k:
            P[[k, pivot_row]] = P[[pivot_row, k]]

        for i in range(k + 1, m):
            L_ik = A[P[i], Q[k]] / pivot_val
            A[P[i], Q[k]] = L_ik  

            for j in range(k + 1, n):
                A[P[i], Q[j]] -= L_ik * A[P[k], Q[j]]

    global _last_paqlu_rank, _last_paqlu_tol
    _last_paqlu_rank = r
    _last_paqlu_tol = TOL

    return P, Q, A

def solve(A, b):
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape

    b_arr = np.asarray(b)
    if b_arr.ndim == 1:
        if b_arr.size != m:
            raise ValueError("b has incompatible shape")
        b_arr = b_arr.reshape(m, 1)
    elif b_arr.ndim == 2:
        if b_arr.shape[0] != m:
            raise ValueError("b has incompatible shape")
    else:
        raise ValueError("b must be one- or two-dimensional")
    k = b_arr.shape[1]

    if m == 0:
        if b_arr.size != 0:
            raise ValueError("inconsistent system: no equations but nonzero b")
        N = np.eye(n, dtype=A.dtype)
        c = np.zeros((n, k), dtype=A.dtype)
        return N, (c.reshape(n) if k == 1 else c)

    if n == 0:
        tol = 1e-6
        if not np.allclose(b_arr, 0, atol=tol):
            raise ValueError("inconsistent system: no variables but nonzero b")
        N = np.zeros((0, 0), dtype=A.dtype)
        c = np.zeros((0, k), dtype=A.dtype)
        return N, (c.reshape(0) if k == 1 else c)

    A_work = np.array(A, copy=True)
    P, Q, A_work = paqlu_decomposition_in_place(A_work)  

    tol = globals().get('_last_paqlu_tol', 1e-6)
    r = globals().get('_last_paqlu_rank', None)
    if r is None:
        r = 0
        lim = min(m, n)
        for i in range(lim):
            if abs(A_work[P[i], Q[i]]) > tol:
                r += 1
            else:
                break

    y = b_arr[P, :].astype(A.dtype, copy=True)
    for i in range(m):
        jmax = min(i, r)  
        if jmax > 0:
            L_row = A_work[P[i], Q[:jmax]]        
            y[i, :] -= L_row @ y[:jmax, :]        

    if r < m:
        if y[r:, :].size and np.max(np.abs(y[r:, :])) > tol:
            raise ValueError("inconsistent system: A x = b has no solution")

    U_piv = A_work[np.ix_(P[:r], Q[:r])]  
    z = np.zeros((r, k), dtype=A.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        if i + 1 < r:
            rhs -= U_piv[i, i + 1 : r] @ z[i + 1 : r, :]
        piv = U_piv[i, i]
        if abs(piv) <= tol:
            raise ValueError("inconsistent system: zero pivot encountered")
        z[i, :] = rhs / piv

    # Particular solution
    x_perm = np.zeros((n, k), dtype=A.dtype)
    x_perm[:r, :] = z
    c = np.zeros((n, k), dtype=A.dtype)
    c[Q, :] = x_perm
    if k == 1:
        c = c.reshape(n)

    # Nullspace 
    f = n - r
    N = np.zeros((n, f), dtype=A.dtype)
    if f > 0:
        U_free = -A_work[np.ix_(P[:r], Q[r:n])]  
        X = np.zeros((r, f), dtype=A.dtype)
        for i in range(r - 1, -1, -1):
            rhs = U_free[i, :].copy()
            if i + 1 < r:
                rhs -= U_piv[i, i + 1 : r] @ X[i + 1 : r, :]
            piv = U_piv[i, i]
            if abs(piv) <= tol:
                raise ValueError("inconsistent system: zero pivot in nullspace computation")
            X[i, :] = rhs / piv

        N_perm = np.zeros((n, f), dtype=A.dtype)
        N_perm[:r, :] = X
        if f > 0:
            N_perm[r : r + f, :] = np.eye(f, dtype=A.dtype)

        N[Q, :] = N_perm

    return N, c
