import numpy as np

def paqlu_decomposition_in_place(A, tol=1e-6):

    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    if not np.issubdtype(A.dtype, np.inexact):
        raise TypeError("A must be float/complex; cannot change dtype in-place")
    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)
    r = 0
    for k in range(min(m, n)):
        S = A[k:, k:]
        if S.size == 0:
            break
        S_abs = np.abs(S)
        max_abs = S_abs.max()
        if max_abs <= tol:
            break
        i_rel, j_rel = np.unravel_index(S_abs.argmax(), S_abs.shape)
        pr = k + i_rel  
        pc = k + j_rel  
        if pr != k:
            A[[k, pr], :] = A[[pr, k], :]
            P[[k, pr]] = P[[pr, k]]
        if pc != k:
            A[:, [k, pc]] = A[:, [pc, k]]
            Q[[k, pc]] = Q[[pc, k]]
        piv = A[k, k]
        for i in range(k+1, m):
            A[i, k] = A[i, k] / piv
        for i in range(k+1, m):
            factor = A[i, k]
            A[i, k+1:n] -= factor * A[k, k+1:n]
        r += 1
    return P, Q, A

def solve(A, b):

    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape

    b_arr = np.asarray(b)
    out_dtype = np.complex128 if (np.iscomplexobj(A) or np.iscomplexobj(b_arr)) else np.float64
    if b_arr.ndim == 1:
        if b_arr.size != m:
            A_work = np.array(A, dtype=out_dtype, copy=True)
            P, Q, A_work = paqlu_decomposition_in_place(A_work, tol=1e-6)
            lim = min(m, n)
            r = 0
            for i in range(lim):
                if np.abs(A_work[i, i]) > 1e-6:
                    r += 1
                else:
                    break
            f = n - r
            N = np.zeros((n, f), dtype=out_dtype)
            if f > 0:
                U_piv = A_work[:r, :r]
                U_free = -A_work[:r, r:n]
                X = np.zeros((r, f), dtype=out_dtype)
                for j in range(r - 1, -1, -1):
                    piv = U_piv[j, j]
                    if np.abs(piv) <= 1e-6:
                        return None, N
                    rhs = U_free[j, :].copy()
                    if j + 1 < r:
                        rhs -= U_piv[j, j+1:r] @ X[j+1:r, :]
                    X[j, :] = rhs / piv
                N_perm = np.zeros((n, f), dtype=out_dtype)
                N_perm[:r, :] = X
                if f > 0:
                    N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
                N[Q, :] = N_perm
            return None, N
        b_arr = b_arr.reshape(m, 1)  
    elif b_arr.ndim == 2:
        if b_arr.shape[0] != m:
            A_work = np.array(A, dtype=out_dtype, copy=True)
            P, Q, A_work = paqlu_decomposition_in_place(A_work, tol=1e-6)
            lim = min(m, n)
            r = 0
            for i in range(lim):
                if np.abs(A_work[i, i]) > 1e-6:
                    r += 1
                else:
                    break
            f = n - r
            N = np.zeros((n, f), dtype=out_dtype)
            if f > 0:
                U_piv = A_work[:r, :r]
                U_free = -A_work[:r, r:n]
                X = np.zeros((r, f), dtype=out_dtype)
                for j in range(r - 1, -1, -1):
                    piv = U_piv[j, j]
                    if np.abs(piv) <= 1e-6:
                        return None, N
                    rhs = U_free[j, :].copy()
                    if j + 1 < r:
                        rhs -= U_piv[j, j+1:r] @ X[j+1:r, :]
                    X[j, :] = rhs / piv
                N_perm = np.zeros((n, f), dtype=out_dtype)
                N_perm[:r, :] = X
                if f > 0:
                    N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
                N[Q, :] = N_perm
            return None, N
    else:
        A_work = np.array(A, dtype=out_dtype, copy=True)
        P, Q, A_work = paqlu_decomposition_in_place(A_work, tol=1e-6)
        lim = min(m, n)
        r = 0
        for i in range(lim):
            if np.abs(A_work[i, i]) > 1e-6:
                r += 1
            else:
                break
        f = n - r
        N = np.zeros((n, f), dtype=out_dtype)
        if f > 0:
            U_piv = A_work[:r, :r]
            U_free = -A_work[:r, r:n]
            X = np.zeros((r, f), dtype=out_dtype)
            for j in range(r - 1, -1, -1):
                piv = U_piv[j, j]
                if np.abs(piv) <= 1e-6:
                    return None, N
                rhs = U_free[j, :].copy()
                if j + 1 < r:
                    rhs -= U_piv[j, j+1:r] @ X[j+1:r, :]
                X[j, :] = rhs / piv
            N_perm = np.zeros((n, f), dtype=out_dtype)
            N_perm[:r, :] = X
            if f > 0:
                N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
            N[Q, :] = N_perm
        return None, N

    k = b_arr.shape[1]  
    tol = 1e-6

    A_work = np.array(A, dtype=out_dtype, copy=True)
    P, Q, A_work = paqlu_decomposition_in_place(A_work, tol=tol)

    lim = min(m, n)
    r = 0
    for i in range(lim):
        if np.abs(A_work[i, i]) > tol:
            r += 1
        else:
            break
    f = n - r  

    y = b_arr[P, :].astype(out_dtype, copy=True)
    for i in range(m):
        jmax = min(i, r)
        if jmax > 0:
            y[i, :] -= A_work[i, :jmax] @ y[:jmax, :]

    if r < m and y[r:, :].size:
        if np.linalg.norm(y[r:, :], ord=np.inf) > tol:
            N = np.zeros((n, f), dtype=out_dtype)
            if f > 0:
                U_piv = A_work[:r, :r]
                U_free = -A_work[:r, r:n]
                X = np.zeros((r, f), dtype=out_dtype)
                for j in range(r - 1, -1, -1):
                    piv = U_piv[j, j]
                    if np.abs(piv) <= tol:
                        return None, N
                    rhs = U_free[j, :].copy()
                    if j + 1 < r:
                        rhs -= U_piv[j, j+1:r] @ X[j+1:r, :]
                    X[j, :] = rhs / piv
                N_perm = np.zeros((n, f), dtype=out_dtype)
                N_perm[:r, :] = X
                if f > 0:
                    N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
                N[Q, :] = N_perm
            return None, N

    U_piv = A_work[:r, :r]
    z = np.zeros((r, k), dtype=out_dtype)
    for i in range(r - 1, -1, -1):
        piv = U_piv[i, i]
        if np.abs(piv) <= tol:
            N = np.zeros((n, f), dtype=out_dtype)
            if f > 0:
                U_free = -A_work[:r, r:n]
                X = np.zeros((r, f), dtype=out_dtype)
                for j in range(r - 1, -1, -1):
                    piv2 = U_piv[j, j]
                    if np.abs(piv2) <= tol:
                        break
                    rhs2 = U_free[j, :].copy()
                    if j + 1 < r:
                        rhs2 -= U_piv[j, j+1:r] @ X[j+1:r, :]
                    X[j, :] = rhs2 / piv2
                N_perm = np.zeros((n, f), dtype=out_dtype)
                N_perm[:r, :] = X
                if f > 0:
                    N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
                N[Q, :] = N_perm
            return None, N
        rhs = y[i, :].copy()
        if i + 1 < r:
            rhs -= U_piv[i, i+1:r] @ z[i+1:r, :]
        z[i, :] = rhs / piv

    x_perm = np.zeros((n, k), dtype=out_dtype)
    x_perm[:r, :] = z            
    c = np.zeros((n, k), dtype=out_dtype)
    c[Q, :] = x_perm             
    if k == 1:
        c = c.reshape(n)         

    N = np.zeros((n, f), dtype=out_dtype)
    if f > 0:
        U_free = -A_work[:r, r:n]         
        X = np.zeros((r, f), dtype=out_dtype)
        for i in range(r - 1, -1, -1):
            piv = U_piv[i, i]
            if np.abs(piv) <= tol:
                return None, N
            rhs = U_free[i, :].copy()
            if i + 1 < r:
                rhs -= U_piv[i, i+1:r] @ X[i+1:r, :]
            X[i, :] = rhs / piv
        N_perm = np.zeros((n, f), dtype=out_dtype)
        N_perm[:r, :] = X
        if f > 0:  
            N_perm[r:r+f, :] = np.eye(f, dtype=out_dtype)
        N[Q, :] = N_perm

    return N,c
