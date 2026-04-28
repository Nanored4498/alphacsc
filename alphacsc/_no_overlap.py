import numpy as np
import numba as nb
import pyfftw

from ._base import BaseZEncoder, BaseDSolver
from .update_d_multi import prox_d

# ================================
# Helper functions for numba types
# ================================


def _float64_r(d=1):
    return nb.types.Array(nb.float64, d, 'C', True)


def _float64_w(d=1):
    return nb.types.Array(nb.float64, d, 'C', False)


def _int32_r(d=1):
    return nb.types.Array(nb.int32, d, 'C', True)


def _int32_w(d=1):
    return nb.types.Array(nb.int32, d, 'C', False)


# ==============================
# JIT functions used by Zencoder
# ==============================


@nb.njit(
    nb.void(_float64_r(2), nb.int32, nb.float64,
            _float64_w(), _int32_w(), _int32_w(), _float64_w()),
    cache=True, nogil=True
)
def _dp_fft(proj, n_times_atom, reg,
            dp, last, atom_index, atom_coeff):
    """
    Computes z for one signal X, using dynamic programming.

    This function uses an array `proj` of pre-computed dot products
    between D and X. It differs from `_dp_prod` which computes those
    dot products on the fly when the atoms are small enough.

    Parameters
    ----------

    Inputs

    proj: array, shape (n_atoms, >= n_times)
        Contains proj[i, t], the dot product between D[i]
        and X[:, t-n_times_atom+1:t+1].
    n_times_atom: int
        The length of an atom in the dictionary
    reg: float
        Regularization parameter in the energy
        E = 0.5 * || X - X_hat ||_2^2 + reg * || z ||_0

    Outputs

    dp: array, shape (n_times+1,)
        Contains dp[t], the best energy achievable for
        the sub-signal X[:, :t]
    last: array, shape (n_times+1,)
        Contains last[t], the ending time of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_index: array, shape (n_times+1,)
        Contains atom_index[t], the atom index of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_coeff: array, shape (n_times+1,)
        Contains atom_index[t], the coefficient of the last activation
        in a best signal encoding for the sub-signal X[:, :t]

    Notes
    -----
    Time Complexity: O(n_times * n_atoms).
    """
    p = proj.shape[0]
    reg *= 2  # We scale the energy by 2 to avoid halving the l2 objective
    for t in range(n_times_atom, len(dp)):
        max_proj, ind = 0, 0
        for i in range(p):
            x = np.abs(proj[i, t-1])
            if x > max_proj:
                max_proj, ind = x, i
        E = dp[t-n_times_atom] + reg - max_proj**2
        if E < dp[t-1]:
            dp[t], last[t] = E, t
            atom_index[t] = ind
            atom_coeff[t] = proj[ind, t-1]
        else:
            dp[t], last[t] = dp[t-1], last[t-1]


@nb.njit(
    nb.void(_float64_r(3), _float64_r(), _float64_r(2), nb.float64,
            _float64_w(), _int32_w(), _int32_w(), _float64_w()),
    cache=True, nogil=True
)
def _dp_prod(D, D_mul, X, reg,
             dp, last, atom_index, atom_coeff):
    """
    Computes z for one signal X, using D, the dictionary.

    This function differs from `_dp_fft` which used precomputed
    dot products using FFT.

    Parameters
    ----------

    Inputs

    D: array, shape (n_atoms, n_channels, n_times_atom)
        The dictionary
    D_mul: array, shape (n_atoms,)
        The inverse of the atom norms
    X: array, shape (n_channels, n_times)
        The signal
    reg: float
        Regularization parameter in the energy
        E = 0.5 * || X - X_hat ||_2^2 + reg * || z ||_0

    Outputs

    dp: array, shape (n_times+1,)
        Contains dp[t], the best energy achievable for
        the sub-signal X[:, :t]
    last: array, shape (n_times+1,)
        Contains last[t], the ending time of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_index: array, shape (n_times+1,)
        Contains atom_index[t], the atom index of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_coeff: array, shape (n_times+1,)
        Contains atom_index[t], the coefficient of the last activation
        in a best signal encoding for the sub-signal X[:, :t]

    Notes
    -----
    Time Complexity: O(n_times * n_atoms * n_channels * n_times_atom).
    """
    p, C, L = D.shape
    reg *= 2  # We scale the energy by 2 to avoid halving the l2 objective
    for t in range(L, len(dp)):
        max_proj, ind = 0, 0
        for i in range(p):
            proj = 0
            for c in range(C):
                proj += D[i, c] @ X[c, t-L:t]
            proj *= D_mul[i]
            if np.abs(proj) > np.abs(max_proj):
                max_proj, ind = proj, i
        E = dp[t-L] + reg - max_proj**2
        if E < dp[t-1]:
            dp[t], last[t] = E, t
            atom_index[t] = ind
            atom_coeff[t] = max_proj
        else:
            dp[t], last[t] = dp[t-1], last[t-1]


@nb.njit(
    nb.int32(_float64_r(), _int32_r(), _int32_r(), _float64_r(), nb.int32,
             _int32_w(2), _float64_w()),
    cache=True, nogil=True
)
def _get_nz_values(D_mul, last, atom_index, atom_coeff, n_times_atom,
                   nz_index, nz_coeff):
    """
    Retrieve z as a sparse representation from the array `last`,
    obtained via dynamic programming (DP).

    Parameters
    ----------

    Inputs

    D_mul: array, shape (n_atoms,)
        The inverse of the atom norms
    last: array, shape (n_times+1,)
        Contains last[t], the ending time of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_index: array, shape (n_times+1,)
        Contains atom_index[t], the atom index of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    atom_coeff: array, shape (n_times+1,)
        Contains atom_index[t], the coefficient of the last activation
        in a best signal encoding for the sub-signal X[:, :t]
    n_times_atom: int
        The length of an atom in the dictionary

    Outputs

    nz_index: array, shape (>= nnz, 2)
        Contains nz_index[i][0], the atom index of the i-th activation
        of the signal encoding, and nz_index[i][1], the start time of
        the i-th activation.
    nz_coeff: array, shape (>= nnz)
        Contains nz_coeff[i], the coefficient of the i-th activation
        of the signal encoding.

    Returns
    -------
    nnz : int
        The number of non zero entries in z.

    Notes
    -----
    Time Complexity: O(nnz), where nnz is the number of non zero entries in z.
    """
    nnz, t = 0, last[-1]
    while t != -1:
        atom = atom_index[t]
        nz_index[nnz][0] = atom
        nz_index[nnz][1] = t-n_times_atom
        nz_coeff[nnz] = atom_coeff[t] * D_mul[atom]
        nnz += 1
        t = last[t-n_times_atom]
    return nnz


@nb.njit(
    nb.float64(_float64_r(3), _float64_r(), _float64_r(3), _int32_r(),
               nb.float64, nb.float64,
               _int32_w(3), _float64_w(2)),
    cache=True, nogil=True
)
def _compute_z_from_T(D, D_mul, X, nnz,
                      XtX, reg,
                      nz_index, nz_coeff):
    """
    Computes the best z value for a gien dictionary D and
    a temporal support T stored in nz_index

    Parameters
    ----------

    Inputs

    D: array, shape (n_atoms, n_channels, n_times_atom)
        The dictionary
    D_mul: array, shape (n_atoms,)
        The inverse of the atom norms
    X: array, shape (n_trials, n_channels, n_times)
        The signal
    nnz: array, shape (n_trials,)
        The number of activations to encode each trial
    XtX: float
        The squared norm of X
    reg: float
        Regularization parameter in the energy
        E = 0.5 * || X - X_hat ||_2^2 + reg * || z ||_0

    Outputs

    nz_index: array, shape (n_trials, >= nnz, 2)
        Contains nz_index[trial,i,0], the atom index of the i-th activation
        of the signal encoding, and nz_index[trial,i,1], the start time of
        the i-th activation for each trial. The atom index is updated by
        this function while the start time is an input left unchanged.
    nz_coeff: array, shape (n_trials, >= nnz)
        Contains nz_coeff[trial,i], the coefficient of the i-th activation
        of the signal encoding of each trial.

    Returns
    -------
    E : float
        The energy/objective associated to the z value computed.

    Notes
    -----
    Time Complexity: O(nnz * n_atoms * n_channels * n_times_atom),
        where nnz is the number of non zero entries in z.
    """
    E0, Ereg = XtX, 0
    N = X.shape[0]
    p, C, L = D.shape
    for trial in range(N):
        Ereg += nnz[trial]
        for ind in range(nnz[trial]):
            t = nz_index[trial][ind][1]
            max_proj, best = 0, 0
            for atom in range(p):
                proj = 0
                for c in range(C):
                    proj += D[atom, c] @ X[trial, c, t:t+L]
                if np.abs(proj) > np.abs(max_proj):
                    max_proj, best = proj, atom
            E0 -= max_proj**2
            nz_index[trial][ind][0] = best
            nz_coeff[trial][ind] = max_proj * D_mul[best]
    return .5 * E0 + reg * Ereg


@nb.njit(
    nb.float64(_float64_r(3), _float64_r(3), _int32_r(), _int32_r(3),
               nb.float64, nb.float64),
    cache=True, nogil=True
)
def _compute_objective(D, X, nnz, nz_index,
                       XtX, reg):
    """
    Computes the objective E for a gien dictionary D and
    a temporal support T stored in nz_index.
    To do so, the best z matching the temporal support T is computed.

    Parameters
    ----------
    D: array, shape (n_atoms, n_channels, n_times_atom)
        The dictionary
    X: array, shape (n_trials, n_channels, n_times)
        The signal
    nnz: array, shape (n_trials,)
        The number of activations to encode each trial
    nz_index: array, shape (n_trials, >= nnz, 2)
        Contains nz_index[trial,i,1], the start time of the i-th activation
        for each trial
    XtX: float
        The squared norm of X
    reg: float
        Regularization parameter in the energy
        E = 0.5 * || X - X_hat ||_2^2 + reg * || z ||_0

    Returns
    -------
    E : float
        The energy/objective associated to the z value computed.

    Notes
    -----
    Time Complexity: O(nnz * n_atoms * n_channels * n_times_atom),
        where nnz is the number of non zero entries in z.
    """
    E0, Ereg = XtX, 0
    N = X.shape[0]
    p, C, L = D.shape
    D2 = D.copy()
    for atom in range(p):
        D2[atom] /= np.linalg.norm(D2[atom])
    for trial in range(N):
        Ereg += nnz[trial]
        for ind in range(nnz[trial]):
            t = nz_index[trial][ind][1]
            max_proj = 0
            for atom in range(p):
                proj = 0
                for c in range(C):
                    proj += D2[atom, c] @ X[trial, c, t:t+L]
                max_proj = max(max_proj, np.abs(proj))
            E0 -= max_proj**2
    return .5 * E0 + reg * Ereg


@nb.njit(
    nb.int64(_int32_r(), _int32_r(3), _float64_r(2),
             _float64_r(3), _float64_r(3)),
    cache=True, nogil=True
)
def _find_max_error_patch(nnz, nz_index, nz_coeff,
                          D, X):
    """
    Find the patch with the maximum residual error.

    Returns
    -------
    max_error_ind : int64
        An integer such that:
        * (max_error_ind >> 32) is the trial index in which the patch
            with max error is found.
        * (max_error_ind & 0xffffffff) is the start time of the patch
            with max error.

    Notes
    -----
    Time Complexity: O(n_trials * n_channels * n_times)
    """

    T = X.shape[-1]
    C, L = D.shape[-2:]
    # patch is a circular array storing the last L=n_times_atom
    # entries of the residual (X - X_hat)
    patch = np.zeros(L, dtype=np.float64)
    max_error = 0.  # The max error seen for a patch
    max_error_ind = 0  # The returned value of this function

    for trial in range(len(nnz)):

        nz_ind = 0  # The index of the next non-zero z entry to be seen
        atom_ind = 0  # The atom index of the last non-zero z entry seen
        atom_coeff = 0.  # The coefficient of the last non-zero z entry seen
        # t0 and t1 are the start and end time
        # of the last non-zero z entry seen
        t0, t1 = 0, 0
        error = 0.  # The error associated to the current patch

        for t in range(T):

            # If z has a non-zero entry at time t, we read it
            if nz_ind < nnz[trial] and nz_index[trial, nz_ind, 1] == t:
                atom_ind = nz_index[trial, nz_ind, 0]
                atom_coeff = nz_coeff[trial, nz_ind]
                t0 = t
                t1 = t+L
                nz_ind += 1

            # Index of the current time in the circular array
            tp = t % L

            # We remove the error associated to time t-L
            # which is non longer part of the current patch
            if t >= L:
                error -= patch[tp]

            # We compute the error associated to time t
            diff = 0
            for c in range(C):
                dc = X[trial, c, t]
                if t < t1:
                    dc -= atom_coeff * D[atom_ind, c, t-t0]
                diff += dc**2
            patch[tp] = diff
            error += diff

            # We update the max error
            if error > max_error:
                max_error = error
                max_error_ind = (trial << 32) | max(0, t-L+1)

    return max_error_ind


@nb.njit(
    nb.void(_int32_r(), _int32_r(3), _float64_r(2), _float64_r(3),
            _float64_w(3), _float64_w(3), _float64_w(3), _int32_w()),
    cache=True, nogil=True
)
def _compute_z_hat(nnz, nz_index, nz_coeff, X,
                   z_hat, ztz, ztX, nnz_atom):
    """
    Computes the dense representation of z_hat from its spartse representation.
    Also computes ztz, ztX and nnz_atom which is the number of non-zero entries
    of z per atom.

    Notes
    -----
    Time Complexity: O(nnz * n_channels * n_times_atom),
        where nnz is the number of non zero entries in z.
    """
    z_hat[:] = 0
    p, C, L = ztX.shape
    t0 = ztz.shape[2]//2
    for atom in range(p):
        ztz[atom, atom, t0] = 0
    ztX[:] = 0
    nnz_atom[:] = 0
    for trial in range(len(nnz)):
        for ind in range(nnz[trial]):
            atom = nz_index[trial, ind, 0]
            t = nz_index[trial, ind, 1]
            coeff = nz_coeff[trial, ind]
            z_hat[trial, atom, t] = coeff
            ztz[atom, atom, t0] += coeff**2
            for c in range(C):
                for dt in range(L):
                    ztX[atom, c, dt] += coeff * X[trial, c, t+dt]
            nnz_atom[atom] += 1


# =============================
# JIT functions used by Dsolver
# =============================

MAX_KMEAN_STEPS = 10


@nb.njit(
    nb.void(_float64_r(3), _int32_r(), _int32_r(3), nb.int32, nb.int32,
            _float64_w(), _int32_w()),
    cache=True, nogil=True
)
def kmean_init(X, nnz, nz_index, n_atoms, n_times_atom,
               init_data, updt_data):
    """
    Computes a partition of the temporal support T in n_atoms parts.
    The algorithm is inspired by k-mean++.

    Notes
    -----
    The algorithm creates a set of n_atoms normalized patches U.
    U is initialized with u_1, the patch of length n_times_atom starting
    at a time in the temporal support T and having the highest l2 norm.
    Then, new patches u_i are added maximizing

    .. math::
        \\| x \\|^2 - \\max_{j < i} (u_j^T x)^2

    for x, a patch of length n_times_atom
    starting at a time in the temporal support T.

    Then, the partition is obtained by associating a time t in T to the
    parition index given by:

    .. math::
        \\argmax_i | u_i^T x[t, t:t+\\text{n_times_atom}] |

    Time Complexity: O(nnz * n_atoms * n_channels * n_times_atom),
        where nnz is the number of non zero entries in z.
    """
    N, C = X.shape[:2]
    S = 0  # Index over the non-zero entries of z

    # Computes the squared l2 norm of all patches starting
    # at a non-zero entry of z.
    for trial in range(N):
        for ind in range(nnz[trial]):
            t = nz_index[trial, ind, 1]
            x_norm2 = 0
            for c in range(C):
                xc = X[trial, c, t:t+n_times_atom]
                x_norm2 += xc @ xc
            init_data[S] = x_norm2
            S += 1

    x2 = init_data[:S]  # squared l2 norm of patches
    dist = init_data[S:2*S]  # squared distance of patches to the set U
    closest = updt_data[:S]  # partition index associated to each patch
    dist[:] = x2
    closest[:] = 0
    # u is a temporary array containing a normalized patch added to U
    u = np.empty((C, n_times_atom), dtype=np.float64)

    for atom in range(n_atoms):
        # Find the farthest patch from U
        farthest = dist.argmax()
        trial = 0  # Index of the trial containing this patch
        while farthest >= nnz[trial]:
            farthest -= nnz[trial]
            trial += 1
        t = nz_index[trial, farthest, 1]  # Start time of the patch
        u[:] = X[trial, :, t:t+n_times_atom]
        u /= np.linalg.norm(u)

        # Updates dist and closest arrays
        s = 0
        for trial in range(N):
            for ind in range(nnz[trial]):
                t = nz_index[trial, ind, 1]
                proj = 0
                for c in range(C):
                    proj += u[c] @ X[trial, c, t:t+n_times_atom]
                d = x2[s] - proj*proj
                if d < dist[s]:
                    dist[s], closest[s] = d, atom
                s += 1


@nb.njit(
    nb.float64(_float64_r(3), _int32_r(), _int32_r(3), nb.int32,
               _int32_w(), _float64_w(2), _float64_w(3)),
    cache=True, nogil=True
)
def kmean(X, nnz, nz_index, S,
          updt_data, Y, D):
    """
    Computes a partition of the temporal support T in len(D) parts.
    The algorithm is inspired by k-mean. If a partition is already stored
    in updt_data, S should be the size of the temporal support. Otherwise,
    S should be 0, and updt_data is initialized with a the parition given
    by z in nz_index.

    Returns
    -------
    E : float
        The energy/objective associated to the dictionary D computed.

    Notes
    -----
    The algorithm creates a dictionary by alternatively computing the best
    dictionary D for a given partition of the temporal support T and then,
    updatating the partition such that each patch starting at a non-zero
    entry of z, is associated to the row of D with the highest dot product.

    Time Complexity: O(nnz * n_channels * n_times_atom
                       * [n_atoms + min(nnz, n_channels * n_times_atom)]),
        where nnz is the number of non zero entries in z.
    """
    N = X.shape[0]
    p, C, L = D.shape

    # When S=0, use the partition of z
    if S == 0:
        for trial in range(N):
            for ind in range(nnz[trial]):
                updt_data[S] = nz_index[trial, ind, 0]
                S += 1

    closest = updt_data[:S]  # partition index associated to each patch
    old_closest = updt_data[S:2*S]  # previous partition for stopping criteria
    cluster_size = updt_data[2*S:2*S+p+1]  # size of clusters in the partition

    for _ in range(MAX_KMEAN_STEPS):
        # Compute the sizes of the clusters
        cluster_size[:] = 0
        for s in range(S):
            cluster_size[closest[s]] += 1
        for atom in range(p):
            cluster_size[atom+1] += cluster_size[atom]

        # Fill Y by packing patches belonging to the same cluster together
        s = 0
        for trial in range(N):
            for ind in range(nnz[trial]):
                t = nz_index[trial, ind, 1]
                atom = closest[s]
                cluster_size[atom] -= 1
                j = cluster_size[atom]
                for c in range(C):
                    Y[j, c*L:c*L+L] = X[trial, c, t:t+L]
                s += 1

        # For each cluster, we use the max singular vector as a row of D
        for atom in range(p):
            i, j = cluster_size[atom], cluster_size[atom+1]
            if i == j:
                D[atom] = 0
                continue
            d = np.linalg.svd(Y[i:j], full_matrices=False)[2][0]
            for c in range(C):
                D[atom, c] = d[c*L:c*L+L]

        # We recompute the partition by assigning each patch to the row
        # of D with the highest dot product
        old_closest, closest = closest, old_closest
        E = 0  # The l2 objective
        s = 0
        for trial in range(N):
            for ind in range(nnz[trial]):
                t = nz_index[trial, ind, 1]
                max_proj, best = 0, 0
                for atom in range(p):
                    proj = 0
                    for c in range(C):
                        proj += D[atom, c] @ X[trial, c, t:t+L]
                    proj = np.abs(proj)
                    if proj < max_proj:
                        continue
                    max_proj, best = proj, atom
                E += max_proj**2
                closest[s] = best
                s += 1

        # If partitions did not change since last iteration, we stop
        if np.array_equal(old_closest, closest):
            break

    return E


# ==============================================
# ZEncoder for L0 regularization without overlap
# ==============================================


class NoOverlapEncoder(BaseZEncoder):
    """
    Zencoder for CDL using L0 regularization without overlap between atoms.
    If z_hat has non zero entries at times t1 and t2,
    then abs(t2 - t1) >= n_times_atom.
    """

    # TODO: Adjust the value of this constant
    USE_FFT_THRESHOLD = 2.

    def __init__(self, X, D_hat, n_jobs, solver_kwargs, reg):
        super().__init__(X, D_hat, n_jobs, solver_kwargs, reg)

        self.dp = np.empty(self.n_times+1, dtype=np.float64)
        self.last = np.empty(self.n_times+1, dtype=np.int32)
        self.atom_index = np.empty(self.n_times+1, dtype=np.int32)
        self.atom_coeff = np.empty(self.n_times+1, dtype=np.float64)
        self.dp[:self.n_times_atom] = 0
        self.last[:self.n_times_atom] = -1

        n_trials = X.shape[0]
        self.use_fft = (
            (n_trials + self.n_channels) * self.n_times_atom
            >
            self.USE_FFT_THRESHOLD * np.log2(self.n_times)
        )
        if self.use_fft:
            T2 = pyfftw.next_fast_len(self.n_times)
            Tc = T2//2+1
            self.X_fft = pyfftw.interfaces.numpy_fft.rfft(self.X, n=T2) / T2
            n_atoms = D_hat.shape[0]
            self.fft_data = pyfftw.zeros_aligned(
                (n_atoms, self.n_channels, 2*Tc),
                dtype='float64'
            )
            self.fft_out = np.ndarray(
                (n_atoms, self.n_channels, Tc),
                dtype='complex128',
                buffer=self.fft_data.data
            )
            self.proj = pyfftw.zeros_aligned((n_atoms, 2*Tc), dtype='float64')
            self.proj_in = np.ndarray((n_atoms, Tc), dtype='complex128',
                                      buffer=self.proj.data)
            self.fft_fwd = pyfftw.FFTW(
                self.fft_data[..., :T2],
                self.fft_out,
                axes=(-1,),
                direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',)
            )
            self.fft_bwd = pyfftw.FFTW(
                self.proj_in,
                self.proj[..., :T2],
                axes=(-1,),
                direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE',),
                normalise_idft=False
            )

        self.nnz = np.zeros(n_trials, dtype=np.int32)
        self.nz_index = np.empty(
            (n_trials, self.n_times // self.n_times_atom, 2),
            dtype=np.int32
        )
        self.nz_coeff = np.empty(
            (n_trials, self.n_times // self.n_times_atom),
            dtype=np.float64
        )
        self.total_nnz = 0
        self.z_hat_computed = False

        self._prox()

    def _prox(self):
        self.D_mul = np.linalg.norm(self.D_hat, axis=(1, 2))
        np.divide(1., self.D_mul, out=self.D_mul, where=self.D_mul != 0)
        self.cost = None

    def compute_z(self):
        self.cost = self.XtX
        self.z_hat_computed = False

        if self.use_fft:
            # TODO: use batches of size 8 instead of size p
            # in order to not allocate too large buffers
            self.fft_data[..., :self.n_times_atom] = (
                self.D_hat[..., ::-1] * self.D_mul[:, None, None]
            )
            self.fft_data[..., self.n_times_atom:] = 0
            self.fft_fwd()
            for trial in range(self.X.shape[0]):
                np.einsum("ict,ct->it", self.fft_out, self.X_fft[trial],
                          out=self.proj_in)
                self.fft_bwd()
                _dp_fft(self.proj, self.n_times_atom, self.reg,
                        self.dp, self.last, self.atom_index, self.atom_coeff)
                self.nnz[trial] = (
                    _get_nz_values(self.D_mul, self.last,
                                   self.atom_index, self.atom_coeff,
                                   self.n_times_atom,
                                   self.nz_index[trial], self.nz_coeff[trial])
                )
                self.cost += self.dp[-1]
        else:
            for trial in range(self.X.shape[0]):
                _dp_prod(self.D_hat, self.D_mul, self.X[trial], self.reg,
                         self.dp, self.last, self.atom_index, self.atom_coeff)
                self.nnz[trial] = (
                    _get_nz_values(self.D_mul, self.last,
                                   self.atom_index, self.atom_coeff,
                                   self.n_times_atom,
                                   self.nz_index[trial], self.nz_coeff[trial])
                )
                self.cost += self.dp[-1]

        self.total_nnz = self.nnz.sum()
        self.cost *= .5

    def compute_objective(self, D):
        return _compute_objective(D, self.X,
                                  self.nnz, self.nz_index,
                                  self.XtX, self.reg)

    def _update_z(self):
        if self.cost is not None:
            return
        self.cost = _compute_z_from_T(self.D_hat, self.D_mul, self.X, self.nnz,
                                      self.XtX, self.reg,
                                      self.nz_index, self.nz_coeff)
        self.z_hat_computed = False

    def get_cost(self):
        self._update_z()
        return self.cost

    def get_max_error_patch(self):
        self._update_z()
        ind = _find_max_error_patch(self.nnz, self.nz_index, self.nz_coeff,
                                    self.D_hat, self.X)
        atom_ind = ind >> 32
        t = ind & ((1 << 32)-1)
        return self.X[atom_ind, :, t:t+self.n_times_atom][None].copy()

    def get_z_sparse(self):
        self._update_z()
        return self.nnz, self.nz_index, self.nz_coeff

    def _compute_dense_z_hat(self):
        self._update_z()
        if self.z_hat_computed:
            return
        if not hasattr(self, "z_hat"):
            self.z_hat = np.empty(self.get_z_hat_shape(), dtype=np.float64)
            n_atoms = self.D_hat.shape[0]
            self.ztz = np.zeros(
                (n_atoms, n_atoms, 2*self.n_times_atom-1),
                dtype=np.float64
            )
            self.ztX = np.empty_like(self.D_hat)
            self.nnz_atom = np.empty(n_atoms, dtype=np.int32)
        _compute_z_hat(self.nnz, self.nz_index, self.nz_coeff, self.X,
                       self.z_hat, self.ztz, self.ztX, self.nnz_atom)
        self.z_hat_computed = True

    def get_z_hat(self):
        self._compute_dense_z_hat()
        return self.z_hat

    def get_z_nnz(self):
        self._compute_dense_z_hat()
        return self.nnz_atom

    def set_D(self, D):
        self.D_hat = D
        self._prox()

    def get_constants(self):
        self._compute_dense_z_hat()
        return super().get_constants()


# =============================================
# Dsolver for L0 regularization without overlap
# =============================================


class NoOverlapDSolver(BaseDSolver):

    def __init__(self, n_channels, n_atoms, n_times_atom,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug):
        super().__init__(n_channels, n_atoms, n_times_atom,
                         uv_constraint, D_init, resample_strategy, window, eps,
                         max_iter, momentum, random_state, verbose, debug)

    def update_D(self, z_encoder):
        nnz, nz_index, _ = z_encoder.get_z_sparse()
        S = nnz.sum()
        if (not hasattr(self, 'Y')) or self.Y.shape[0] < S:
            N, _, T = z_encoder.X.shape
            S2 = min(int(1.125*S), N * (T // self.n_times_atom))
            self.Y = np.empty((S2, self.n_channels * self.n_times_atom),
                              dtype=np.float64)
            self.kmean_init_data = np.empty(2*S2, dtype=np.float64)
            self.kmean_updt_data = np.empty(2*S2 + self.n_atoms + 1,
                                            dtype=np.int32)
            self.D_tmp = np.empty_like(self.D_hat)
        E0 = kmean(z_encoder.X, nnz, nz_index, 0,
                   self.kmean_updt_data, self.Y, self.D_tmp)
        kmean_init(z_encoder.X, nnz, nz_index, self.n_atoms, self.n_times_atom,
                   self.kmean_init_data, self.kmean_updt_data)
        E1 = kmean(z_encoder.X, nnz, nz_index, S,
                   self.kmean_updt_data, self.Y, self.D_hat)
        if E0 > E1:
            self.D_tmp, self.D_hat = self.D_hat, self.D_tmp
        z_encoder.set_D(self.D_hat)
        return self.D_hat

    def prox(self, D_hat):
        return prox_d(D_hat)
