import numpy as np
import numba as nb
import pyfftw

from ._base_solver import BaseZEncoder, BaseDSolver
from .update_d_multi import prox_d

def _float64_r(d=1):
    return nb.types.Array(nb.float64, d, 'C', True)
def _float64_w(d=1):
    return nb.types.Array(nb.float64, d, 'C', False)
def _int32_r(d=1):
    return nb.types.Array(nb.int32, d, 'C', True)
def _int32_w(d=1):
    return nb.types.Array(nb.int32, d, 'C', False)

@nb.njit(
    nb.void(
        _float64_r(2), nb.int32, nb.float64,
        _float64_w(), _int32_w(), _int32_w(), _float64_w()
    ), cache=True, nogil=True
)
def _dp_updt_fft(proj, L, penalty,
                 dp, last, atom_index, atom_coeff):
    p = proj.shape[0]
    penalty *= 2
    for t in range(L, len(dp)):
        max_proj, ind = 0, 0
        for i in range(p):
            x = np.abs(proj[i, t-1])
            if x > max_proj:
                max_proj, ind = x, i
        E = dp[t-L] + penalty - max_proj**2
        if E < dp[t-1]:
            dp[t], last[t] = E, t
            atom_index[t] = ind
            atom_coeff[t] = proj[ind, t-1]
        else:
            dp[t], last[t] = dp[t-1], last[t-1]


@nb.njit(
    nb.void(_float64_r(3), _float64_r(), _float64_r(2), nb.float64,
            _float64_w(), _int32_w(), _int32_w(), _float64_w()
    ), cache=True, nogil=True
)
def _dp_updt_prod(D, D_mul, X, penalty,
                  dp, last, atom_index, atom_coeff):
    p, chan, L = D.shape
    penalty *= 2
    for t in range(L, len(dp)):
        max_proj, ind = 0, 0
        for i in range(p):
            proj = 0
            for c in range(chan):
                proj += D[i,c] @ X[c,t-L:t]
            proj *= D_mul[i]
            if np.abs(proj) > np.abs(max_proj):
                max_proj, ind = proj, i
        E = dp[t-L] + penalty - max_proj**2
        if E < dp[t-1]:
            dp[t], last[t] = E, t
            atom_index[t] = ind
            atom_coeff[t] = max_proj
        else:
            dp[t], last[t] = dp[t-1], last[t-1]


@nb.njit(nb.int32(_float64_r(), _int32_r(), _int32_r(), _float64_r(), nb.int32, _int32_w(2), _float64_w()), cache=True, nogil=True)
def _get_nz_values(D_mul, last, atom_index, atom_coeff, L, nz_index, nz_coeff):
    k, t = 0, last[-1]
    while t != -1:
        atom = atom_index[t]
        nz_index[k][0] = atom
        nz_index[k][1] = t-L
        nz_coeff[k] = atom_coeff[t] * D_mul[atom]
        k += 1
        t = last[t-L]
    return k

@nb.njit(
    nb.float64(
        _float64_r(3), _float64_r(3), _int32_r(), _int32_r(3), _float64_r(2),
        nb.float64, nb.float64
    ), cache=True, nogil=True
)
def _compute_objective(D, X, nnz, nz_index, nz_coeff,
                       XtX, penalty):
    E0 = XtX
    Ereg = 0
    N = X.shape[0]
    p, C, L = D.shape
    D2 = np.empty(p)
    for i in range(p):
        D2[i] = D[i].ravel() @ D[i].ravel()
    for trial in range(N):
        Ereg += nnz[trial]
        for i in range(nnz[trial]):
            ind = nz_index[trial][i][0]
            t = nz_index[trial][i][1]
            coeff = nz_coeff[trial][i]
            proj = 0
            for c in range(C):
                proj += D[ind, c] @ X[trial, c, t:t+L]
            E0 += coeff * (coeff * D2[ind] - 2. * proj)
    return .5 * E0 + penalty * Ereg

@nb.njit(nb.int64(_int32_r(), _int32_r(3), _float64_r(2), _float64_r(3), _float64_r(3)), cache=True, nogil=True)
def _find_max_error_patch(nnz, nz_index, nz_coeff, D, X):
    T = X.shape[-1]
    L = D.shape[-1]
    C = X.shape[1]
    patch = np.zeros(L, dtype=np.float64)
    max_error = 0.
    max_error_ind = 0
    for trial in range(len(nnz)):
        nz_ind = 0
        atom_ind = 0
        atom_coeff = 0.
        t0, t1 = 0, 0
        error = 0.
        for t in range(T):
            if nz_ind < nnz[trial] and nz_index[trial, nz_ind, 1] == t:
                atom_ind = nz_index[trial, nz_ind, 0]
                atom_coeff = nz_coeff[trial, nz_ind]
                t0 = t
                t1 = t+L
                nz_ind += 1
            tp = t%L
            if t >= L: error -= patch[tp]
            diff = 0
            for c in range(C):
                dc = X[trial, c, t]
                if t < t1: diff -= atom_coeff * D[atom_ind, c, t-t0]
                diff += dc**2
            patch[tp] = diff
            error += diff
            if error > max_error:
                max_error = error
                max_error_ind = (trial<<32) | max(0, t-L+1)
    return max_error_ind

@nb.njit(
    nb.void(
        _int32_r(), _int32_r(3), _float64_r(2), _float64_r(3),
        _float64_w(3), _float64_w(3), _float64_w(3), _int32_w()
    ), cache=True, nogil=True
)
def _compute_z_hat(nnz, nz_index, nz_coeff, X,
                   z_hat, ztz, ztX, nnz_atom):
    z_hat[:] = 0
    p = ztz.shape[0]
    t0 = ztz.shape[2]//2
    for i in range(p):
        ztz[i, i, t0] = 0
    L = ztX.shape[2]
    ztX[:] = 0
    nnz_atom[:] = 0
    for trial in range(len(nnz)):
        for i in range(nnz[trial]):
            ind = nz_index[trial, i, 0]
            t = nz_index[trial, i, 1]
            coeff = nz_coeff[trial, i]
            z_hat[trial, ind, t] = coeff
            ztz[ind, ind, t0] += coeff**2
            ztX[trial] += X[trial, :, t:t+L]
            nnz_atom[ind] += 1

MAX_KMEAN_STEPS = 10

@nb.njit(
    nb.void(
        _float64_r(3), _int32_r(), _int32_r(3), _float64_w(), _int32_w(), _float64_w(2), _float64_w(3)
    ), cache=True, nogil=True
)
def kmean(X, nnz, nz_index, init_data, updt_data, Y, D):
    #init
    N = len(nnz)
    p, C, L = D.shape
    S = 0
    for trial in range(N):
        x = X[trial]
        for ind in range(nnz[trial]):
            t = nz_index[trial, ind, 1]
            n2seg = 0
            for c in range(C):
                n2seg += x[c, t:t+L] @ x[c, t:t+L]
            init_data[S] = n2seg
            S += 1
    y2 = init_data[:S]
    dist = init_data[S:2*S]
    closest = updt_data[:S]
    old_closest = updt_data[S:2*S]
    dist[:] = y2
    closest[:] = 0
    u = np.empty((C, L), dtype=np.float64)
    for atom in range(p):
        farthest = dist[:S].argmax()
        trial = 0
        while farthest >= nnz[trial]:
            farthest -= nnz[trial]
            trial += 1
        t = nz_index[trial, farthest, 1]
        u[:] = X[trial, :, t:t+L]
        u /= np.linalg.norm(u)
        s = 0
        for trial in range(N):
            for ind in range(nnz[trial]):
                t = nz_index[trial, ind, 1]
                proj = 0
                for c in range(C):
                    proj += u[c] @ X[trial, c, t:t+L]
                d = y2[s] - proj*proj
                if d < dist[s]:
                    dist[s], closest[s] = d, atom
                s += 1
    # updates
    cluster_size = np.empty(p+1, dtype=np.int32)
    for _ in range(MAX_KMEAN_STEPS):
        cluster_size[:p] = 0
        cluster_size[p] = S
        for s in range(S): cluster_size[closest[s]] += 1
        for atom in range(1, p): cluster_size[atom] += cluster_size[atom-1]
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
        for atom in range(p):
            i, j = cluster_size[atom], cluster_size[atom+1]
            if i == j:
                D[atom] = 0
                continue
            d = np.linalg.svd(Y[i:j], full_matrices=False)[2][0]
            for c in range(C):
                D[atom, c] = d[c*L:c*L+L]
        old_closest, closest = closest, old_closest
        s = 0
        for trial in range(N):
            for ind in range(nnz[trial]):
                t = nz_index[trial, ind, 1]
                max_proj, best = 0, 0
                for atom in range(p):
                    proj = 0
                    for c in range(C):
                        proj += D[atom,c] @ X[trial, c, t:t+L]
                    proj = np.abs(proj)
                    if proj < max_proj: continue
                    max_proj, best = proj, atom
                closest[s] = best
                s += 1
        if np.array_equal(old_closest, closest): break

@nb.njit(
    nb.void(
        _float64_r(3), _int32_r(), _int32_r(3), _float64_r(2), _float64_w(3)
    ), cache=True, nogil=True
)
def update_D(X, nnz, nz_index, nz_coeff, D):
    N = len(nnz)
    p, _, L = D.shape
    D[:] = 0
    for trial in range(N):
        for ind in range(nnz[trial]):
            atom = nz_index[trial, ind, 0]
            t = nz_index[trial, ind, 1]
            coeff = nz_coeff[trial, ind]
            D[atom] += coeff * X[trial, :, t:t+L]
    for atom in range(p):
        d2 = D[atom].ravel() @ D[atom].ravel()
        if d2 < 1e-12:
            D[atom] = 0
            continue
        D[atom] /= np.sqrt(d2)

class NoOverlapEncoder(BaseZEncoder):
    # TODO: Adjust the value of this constant
    USE_FFT_THRESHOLD = 2.

    def __init__(self, X, D_hat, n_atoms, n_times_atom, n_jobs, solver_kwargs, reg):
        super().__init__(X, D_hat, n_atoms, n_times_atom, n_jobs, solver_kwargs, reg)

        self.dp = np.empty(self.n_times+1, dtype=np.float64)
        self.last = np.empty(self.n_times+1, dtype=np.int32)
        self.atom_index = np.empty(self.n_times+1, dtype=np.int32)
        self.atom_coeff = np.empty(self.n_times+1, dtype=np.float64)
        self.dp[:self.n_times_atom] = 0
        self.last[:self.n_times_atom] = -1

        self.use_fft = (self.n_trials + self.n_channels) * self.n_times_atom > self.USE_FFT_THRESHOLD * np.log2(self.n_times)
        if self.use_fft:
            T2 = pyfftw.next_fast_len(self.n_times)
            Tc = T2//2+1
            self.X_fft = pyfftw.interfaces.numpy_fft.rfft(self.X, n=T2) / T2
            self.fft_data = pyfftw.zeros_aligned((self.n_atoms, self.n_channels, 2*Tc), dtype='float64')
            self.fft_out = np.ndarray((self.n_atoms, self.n_channels, Tc), dtype='complex128', buffer=self.fft_data.data)
            self.proj = pyfftw.zeros_aligned((self.n_atoms, 2*Tc), dtype='float64')
            self.proj_in = np.ndarray((self.n_atoms, Tc), dtype='complex128', buffer=self.proj.data)
            self.fft_fwd = pyfftw.FFTW(
                self.fft_data[...,:T2],
                self.fft_out,
                axes=(-1,), direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',)
            )
            self.fft_bwd = pyfftw.FFTW(
                self.proj_in,
                self.proj[...,:T2],
                axes=(-1,), direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE',), normalise_idft=False
            )
        
        self.nnz = np.zeros(self.n_trials, dtype=np.int32)
        self.nz_index = np.empty((self.n_trials, self.n_times // self.n_times_atom, 2), dtype=np.int32)
        self.nz_coeff = np.empty((self.n_trials, self.n_times // self.n_times_atom), dtype=np.float64)
        self.total_nnz = 0

        self.z_hat = None
        self.nnz_atom = None
        self.z_hat_computed = False

        self.D_mul = 1. / np.linalg.norm(self.D_hat, axis=(1, 2))
        self.cost = None

    def compute_z(self):
        self.cost = self.XtX
        self.z_hat_computed = False

        if self.use_fft:
            # TODO: use batches of size 8 instead of size p
            # in order to not allocate too large buffers
            self.fft_data[...,:self.n_times_atom] = self.D_hat[...,::-1] * self.D_mul[:, None, None]
            self.fft_data[...,self.n_times_atom:] = 0
            self.fft_fwd()
            for trial in range(self.n_trials):
                np.einsum("ict,ct->it", self.fft_out, self.X_fft[trial], out=self.proj_in)
                self.fft_bwd()
                _dp_updt_fft(self.proj, self.n_times_atom, self.reg, self.dp, self.last, self.atom_index, self.atom_coeff)
                self.nnz[trial] = _get_nz_values(self.D_mul, self.last, self.atom_index, self.atom_coeff, self.n_times_atom,
                                                 self.nz_index[trial], self.nz_coeff[trial])
                self.cost += self.dp[-1]
        else:
            for trial in range(self.n_trials):
                _dp_updt_prod(self.D_hat, self.D_mul, self.X[trial], self.reg,
                              self.dp, self.last, self.atom_index, self.atom_coeff)
                self.nnz[trial] = _get_nz_values(self.D_mul, self.last, self.atom_index, self.atom_coeff, self.n_times_atom,
                                                 self.nz_index[trial], self.nz_coeff[trial])
                self.cost += self.dp[-1]

        self.total_nnz = self.nnz.sum()
        self.cost *= .5

    def compute_objective(self, D):
        return _compute_objective(D, self.X, self.nnz, self.nz_index, self.nz_coeff, self.XtX, self.reg)

    def get_cost(self):
        if self.cost is None:
            self.cost = _compute_objective(self.D_hat, self.X, self.nnz, self.nz_index, self.nz_coeff, self.XtX, self.reg)
        return self.cost

    def get_max_error_patch(self):
        ind = _find_max_error_patch(self.nnz, self.nz_index, self.nz_coeff, self.D_hat, self.X)
        atom_ind = ind >> 32
        t = ind & ((1<<32)-1)
        return self.X[atom_ind, :, t:t+self.n_times_atom][None].copy()

    def get_z_sparse(self):
         return self.nnz, self.nz_index, self.nz_coeff

    def _compute_dense_z_hat(self):
        if not self.z_hat_computed:
            if self.z_hat is None:
                self.z_hat = np.empty(self.get_z_hat_shape(), dtype=np.float64)
                self.ztz = np.zeros((self.n_atoms, self.n_atoms, 2*self.n_times_atom-1), dtype=np.float64)
                self.ztX = np.empty((self.n_trials, self.n_channels, self.n_times_atom), dtype=np.float64)
                self.nnz_atom = np.empty(self.n_atoms, dtype=np.int32)
            _compute_z_hat(self.nnz, self.nz_index, self.nz_coeff, self.X, self.z_hat, self.ztz, self.ztX, self.nnz_atom)
            self.z_hat_computed = True

    def get_z_hat(self):
        self._compute_dense_z_hat()
        return self.z_hat

    def get_z_nnz(self):
        self._compute_dense_z_hat()
        return self.nnz_atom

    def set_D(self, D):
        self.D_hat = D
        self.D_mul = np.linalg.norm(self.D_hat, axis=(1, 2))
        np.divide(1., self.D_mul, out=self.D_mul, where=self.D_mul!=0)
        self.cost = None

    def get_constants(self):
        self._compute_dense_z_hat()
        return super().get_constants()


class NoOverlapSolver(BaseDSolver):

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug):
        super().__init__(n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug)
        
        self.D_hat = np.empty((n_atoms, n_channels, n_times_atom))
        self.Y = None

    def update_D(self, z_encoder, reorder=False):
        nnz, nz_index, nz_coeff = z_encoder.get_z_sparse()
        if reorder:
            S = nnz.sum()
            if self.Y is None or self.Y.shape[0] < S:
                N, _, T = z_encoder.X.shape
                S = min(int(1.125*S), N * (T // self.n_times_atom))
                self.Y = np.empty((S, self.n_channels * self.n_times_atom), dtype=np.float64)
                self.kmean_init_data = np.empty(2*S, dtype=np.float64)
                self.kmean_updt_data = np.empty(2*S, dtype=np.int32)
            kmean(z_encoder.X, nnz, nz_index, self.kmean_init_data, self.kmean_updt_data, self.Y, self.D_hat)
        else:
            update_D(z_encoder.X, nnz, nz_index, nz_coeff, self.D_hat)
        z_encoder.set_D(self.D_hat)
        return self.D_hat

    def prox(self, D_hat):
        return prox_d(D_hat)
