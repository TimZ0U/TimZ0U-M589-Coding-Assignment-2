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

    r = 0
    eps = np.finfo(A.real.dtype).eps  # works for real/complex

    for k in range(min(m, n)):
        # Working submatrix (permuted view): rows P[k:], cols Q[k:]
        # We need its max magnitude to set tol and select the pivot.
        S_abs = np.abs(A[np.ix_(P[k:], Q[k:])])
        if S_abs.size == 0:
            break
        max_abs = S_abs.max()

        # Adaptive tolerance per spec
        tol = max(m, n) * eps * max_abs

        # If the whole remaining block is (numerically) zero ⇒ done (rank = r)
        if max_abs <= tol:
            break

        # Choose pivot anywhere in the working block (full pivoting over the block)
        i_rel, j_rel = np.unravel_index(S_abs.argmax(), S_abs.shape)
        pivot_row = k + i_rel
        pivot_col = k + j_rel

        # Bring pivot to (k,k) in the permuted view by updating P and Q
        if pivot_row != k:
            P[[k, pivot_row]] = P[[pivot_row, k]]
        if pivot_col != k:
            Q[[k, pivot_col]] = Q[[pivot_col, k]]

        pivot = A[P[k], Q[k]]
        # Safety: re-check the chosen pivot vs tol
        if np.abs(pivot) <= tol:
            break

        r += 1

        # Eliminate below the pivot in the current pivot column (permuted view)
        for i in range(k + 1, m):
            L_ik = A[P[i], Q[k]] / pivot
            A[P[i], Q[k]] = L_ik  # store L multiplier in the (strict) lower part
            # Update trailing part of row i
            for j in range(k + 1, n):
                A[P[i], Q[j]] -= L_ik * A[P[k], Q[j]]

    # Expose rank/tolerance for downstream routines / tests
    global _last_paqlu_rank, _last_paqlu_tol
    _last_paqlu_rank = r
    _last_paqlu_tol = tol if 'tol' in locals() else 0.0

    return P, Q, A


def solve(A, b):
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape

    # Shape/convert b
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

    # Edge: m == 0 (no equations) ⇒ any x works; choose c = 0, N = I
    if m == 0:
        if b_arr.size != 0:
            raise ValueError("inconsistent system: no equations but nonzero b")
        N = np.eye(n, dtype=A.dtype)
        c = np.zeros((n, k), dtype=A.dtype)
        return N, (c.reshape(n) if k == 1 else c)

    # Edge: n == 0 (no variables) ⇒ only consistent if b ≈ 0
    if n == 0:
        eps0 = np.finfo(A.real.dtype).eps
        scaleb0 = np.max(np.abs(b_arr)) if b_arr.size else 1.0
        tol0 = max(m, n) * eps0 * max(1.0, scaleb0)
        if b_arr.size and np.linalg.norm(b_arr, ord=np.inf) > tol0:
            raise ValueError("inconsistent system: no variables but nonzero b")
        N = np.zeros((0, 0), dtype=A.dtype)
        c = np.zeros((0, k), dtype=A.dtype)
        return N, (c.reshape(0) if k == 1 else c)

    # Work on a copy for in-place PAQ=LU
    A_work = np.array(A, copy=True)
    P, Q, A_work = paqlu_decomposition_in_place(A_work)

    # Robust solve-stage tolerance (don’t rely solely on _last_paqlu_tol)
    eps = np.finfo(A.real.dtype).eps
    scaleA = np.max(np.abs(A_work)) if A_work.size else 1.0
    scaleb = np.max(np.abs(b_arr)) if b_arr.size else 1.0
    tol_solve = max(m, n) * eps * max(scaleA, scaleb)
    tol = max(globals().get('_last_paqlu_tol', 0.0), tol_solve)

    # Rank (prefer what PAQ computed; otherwise fallback using tol)
    r = globals().get('_last_paqlu_rank', None)
    if r is None:
        r = 0
        lim = min(m, n)
        for i in range(lim):
            if np.abs(A_work[P[i], Q[i]]) > tol:
                r += 1
            else:
                break

    # Forward substitution: L has unit diagonal; multipliers stored below
    y = b_arr[P, :].astype(A.dtype, copy=True)
    for i in range(m):
        jmax = min(i, r)  # only the first r columns are pivots
        if jmax > 0:
            L_row = A_work[P[i], Q[:jmax]]          # shape (jmax,)
            y[i, :] -= L_row @ y[:jmax, :]          # subtract known part

    # Inconsistency check: zero rows (below r) must produce zero RHS
    if r < m and y[r:, :].size:
        if np.linalg.norm(y[r:, :], ord=np.inf) > tol:
            raise ValueError("inconsistent system: A x = b has no solution")

    # Back substitution on U (pivot block)
    U_piv = A_work[np.ix_(P[:r], Q[:r])]            # r x r upper-triangular
    z = np.zeros((r, k), dtype=A.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        if i + 1 < r:
            rhs -= U_piv[i, i + 1 : r] @ z[i + 1 : r, :]
        piv = U_piv[i, i]
        if np.abs(piv) <= tol:
            raise ValueError("inconsistent system: zero pivot encountered")
        z[i, :] = rhs / piv

    # Particular solution in original column order
    x_perm = np.zeros((n, k), dtype=A.dtype)
    x_perm[:r, :] = z                               # pivot vars
    c = np.zeros((n, k), dtype=A.dtype)
    c[Q, :] = x_perm                                # inverse permute by Q
    if k == 1:
        c = c.reshape(n)

    # Nullspace basis N (size n x f, with f = n - r)
    f = n - r
    N = np.zeros((n, f), dtype=A.dtype)
    if f > 0:
        # U_piv X = -U_free  ⇒ X via back substitution
        U_free = -A_work[np.ix_(P[:r], Q[r:n])]     # r x f
        X = np.zeros((r, f), dtype=A.dtype)
        for i in range(r - 1, -1, -1):
            rhs = U_free[i, :].copy()
            if i + 1 < r:
                rhs -= U_piv[i, i + 1 : r] @ X[i + 1 : r, :]
            piv = U_piv[i, i]
            if np.abs(piv) <= tol:
                raise ValueError("inconsistent system: zero pivot in nullspace computation")
            X[i, :] = rhs / piv

        # Stack [X; I_f] in permuted variable order, then inverse-permute by Q
        N_perm = np.zeros((n, f), dtype=A.dtype)
        N_perm[:r, :] = X
        N_perm[r : r + f, :] = np.eye(f, dtype=A.dtype)
        N[Q, :] = N_perm

    return N, c
