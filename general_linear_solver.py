import numpy as np

# --- PAQ = LU with full pivoting, IN-PLACE on A ---
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

        i_rel, j_rel = divmod(np.nanargmax(np.abs(S)), S.shape[1])
        i = k + i_rel
        j = k + j_rel

        if np.abs(A[i, j]) <= tol:
            break

        # row swap
        if i != k:
            A[[k, i], :] = A[[i, k], :]
            P[[k, i]] = P[[i, k]]

        # column swap
        if j != k:
            A[:, [k, j]] = A[:, [j, k]]
            Q[[k, j]] = Q[[j, k]]

        # eliminate below pivot in column k
        piv = A[k, k]
        for i2 in range(k + 1, m):
            A[i2, k] = A[i2, k] / piv
            A[i2, k + 1 :] -= A[i2, k] * A[k, k + 1 :]

        r += 1

    return P, Q, r


def solve(A, b, tol=1e-6):
    """
    Solve Ax = b in the parametric form x = c + N*y_free.

    Returns (c, N) with the following conventions (per instructor update):
      - c : one particular solution (shape (n,) for 1 RHS or (n,k) for k RHS)
      - N : nullspace basis matrix of A (shape (n, nullity))
    If the system is inconsistent, returns (None, N).

    Args:
      A : (m,n) ndarray
      b : (m,) or (m,k) array-like
      tol : pivot/consistency tolerance (default 1e-6, float64)

    Notes:
      - Uses in-place PAQ=LU with full pivoting on a working copy of A.
      - Permutation vectors P, Q follow the convention that A_work is the
        permuted matrix; a vector in the permuted variable order is v_perm
        and the original-order vector is recovered by v[Q] = v_perm.
    """
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

    # dtype policy
    out_dtype = np.complex128 if (np.iscomplexobj(A) or np.iscomplexobj(b_arr)) else np.float64

    # --- edge cases ---
    if m == 0:
        # no equations: any x works -> c = 0, N = I_n
        N = np.eye(int(n), dtype=out_dtype)
        c = np.zeros((int(n), k), dtype=out_dtype)
        return (c.reshape(n) if k == 1 else c), N

    if n == 0:
        # no variables: Ax=b feasible only if b==0
        if b_arr.size and np.linalg.norm(b_arr, ord=np.inf) > tol:
            N = np.zeros((0, 0), dtype=out_dtype)
            return None, N
        N = np.zeros((0, 0), dtype=out_dtype)
        c = np.zeros((0, k), dtype=out_dtype)
        return (c.reshape(0) if k == 1 else c), N

    # --- PAQ=LU on a work copy of A ---
    A_work = A.astype(out_dtype, copy=True)
    P, Q, r = paqlu_decomposition_in_place(A_work, tol=tol)

    # --- Nullspace basis N (depends only on A) ---
    f = int(n - r)
    N = np.zeros((n, f), dtype=out_dtype)
    if f > 0:
        U_piv = A_work[:r, :r]          # r x r (upper)
        U_free = -A_work[:r, r:n]       # r x f  (negative for U X = -U_free)

        # solve U_piv X = -U_free by back substitution
        X = np.zeros((r, f), dtype=out_dtype)
        for i in range(r - 1, -1, -1):
            piv = U_piv[i, i]
            # (piv should be > tol because we counted it in r)
            rhs = U_free[i, :].copy()
            if i + 1 < r:
                rhs -= U_piv[i, i + 1 : r] @ X[i + 1 : r, :]
            X[i, :] = rhs / piv

        N_perm = np.zeros((n, f), dtype=out_dtype)
        N_perm[:r, :] = X
        N_perm[r:, :] = np.eye(f, dtype=out_dtype)
        # undo column permutation: original order rows indexed by Q
        N[Q, :] = N_perm  # shape (n, f)

    # --- Forward substitution: L y = P b (L has unit diagonal, multipliers below) ---
    y = b_arr[P, :].astype(out_dtype, copy=True)
    for i in range(m):
        jmax = min(i, r)
        if jmax > 0:
            y[i, :] -= A_work[i, :jmax] @ y[:jmax, :]

    # --- Consistency check: rows below r must be ~0 ---
    if r < m and y[r:, :].size and np.linalg.norm(y[r:, :], ord=np.inf) > tol:
        # inconsistent: return (None, N) by convention
        return None, N

    # --- Back substitution: U z = y[:r] ---
    z = np.zeros((r, k), dtype=out_dtype)
    if r > 0:
        for i in range(r - 1, -1, -1):
            piv = A_work[i, i]
            rhs = y[i, :].copy()
            if i + 1 < r:
                rhs -= A_work[i, i + 1 : r] @ z[i + 1 : r, :]
            z[i, :] = rhs / piv

    # --- Put basic vars first (perm order), free vars = 0, then undo permutation Q ---
    x_perm = np.zeros((n, k), dtype=out_dtype)
    x_perm[:r, :] = z
    c = np.zeros((n, k), dtype=out_dtype)
    c[Q, :] = x_perm

    # return vector for single RHS
    c = c.reshape(n) if k == 1 else c
    return c, N
