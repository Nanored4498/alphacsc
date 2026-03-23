import numpy as np
import numba as nb

from ._z_encoder import BaseZEncoder
import pyfftw

def _float64_r(d=1):
	return nb.types.Array(nb.float64, d, 'C', True)
def _float64_w(d=1):
	return nb.types.Array(nb.float64, d, 'C', False)
def _int32_r(d=1):
	return nb.types.Array(nb.int32, d, 'C', True)
def _int32_w(d=1):
	return nb.types.Array(nb.int32, d, 'C', False)

@nb.njit(nb.void(_float64_r(2), nb.int32, nb.float64, _float64_w(), _int32_w(), _int32_w(), _float64_w()), cache=True, nogil=True)
def _dp_updt_fft(proj, L, penalty, dp, last, atom_index, atom_coeff):
	p = proj.shape[0]
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


@nb.njit(nb.void(_float64_r(3), _float64_r(2), nb.float64, _float64_w(), _int32_w(), _int32_w(), _float64_w()), cache=True, nogil=True)
def _dp_updt_prod(D, X, penalty, dp, last, atom_index, atom_coeff):
	p, chan, L = D.shape
	for t in range(L, len(dp)):
		max_proj, ind = 0, 0
		for i in range(p):
			proj = 0
			for c in range(chan):
				proj += np.dot(D[i,c], X[c,t-L:t])
			if np.abs(proj) > np.abs(max_proj):
				max_proj, ind = proj, i
		E = dp[t-L] + penalty - max_proj**2
		if E < dp[t-1]:
			dp[t], last[t] = E, t
			atom_index[t] = ind
			atom_coeff[t] = max_proj
		else:
			dp[t], last[t] = dp[t-1], last[t-1]


@nb.njit(nb.int32(_int32_r(), _int32_r(), _int32_r(), nb.int32, _int32_w(2), _float64_w()), cache=True, nogil=True)
def _get_nz_values(last, atom_index, atom_coeff, L, nz_index, nz_coeff):
	k, t = 0, last[-1]
	while t != -1:
		nz_index[k][0] = atom_index[t]
		nz_index[k][1] = t-L
		nz_coeff[k] = atom_coeff[t]
		k += 1
		t = last[t-L]
	return k

@nb.njit(nb.float64(_float64_r(3), _float64_r(3), _int32_r(), _int32_r(3), _float64_r(2)), cache=True, nogil=True)
def _compute_objective(D, X, nnz, nz_index, nz_coeff):
	E = 0
	N, C, _ = X.shape
	L = D.shape[2]
	for trial in range(N):
		E += nnz[trial]
		for i in range(nnz[trial]):
			ind = nz_index[trial][i][0]
			t = nz_index[trial][i][1]
			coeff = nz_coeff[trial][i]
			proj = 0
			for c in range(C):
				proj += np.dot(D[ind, c], X[trial, c, t:t+L])
			E += coeff * (coeff - 2. * proj)
	return E

@nb.njit(nb.void(_int32_r(), _int32_r(3), _float64_r(2), _float64_w(3), _int32_r()), cache=True, nogil=True)
def _compute_z_hat(nnz, nz_index, nz_coeff, z_hat, nnz_atom):
	z_hat[:] = 0
	nnz_atom[:] = 0
	for trial in range(len(nnz)):
		for i in range(nnz[trial]):
			ind = nz_index[trial][i][0]
			t = nz_index[trial][i][1]
			z_hat[trial][ind][t] = nz_coeff[trial][i]
			nnz_atom[ind] += 1


class NoOverlapEncoder(BaseZEncoder):

	MAX_KMEAN_STEPS = 10
	#TODO: Adjust the value of this constant
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
			self.fft_data = pyfftw.zeros_aligned((self.n_atoms, self.self.n_channels, 2*Tc), dtype='float64')
			self.fft_out = np.ndarray((self.n_atoms, self.self.n_channels, Tc), dtype='complex128', buffer=self.fft_data.data)
			self.proj = pyfftw.zeros_aligned((self.n_atoms, 2*Tc), dtype='float64')
			self.proj_in = np.ndarray((self.n_atoms, Tc), dtype='complex128', buffer=self.proj.data)
			self.fft_fwd = pyfftw.FFTW(
				self.fft_data[:,:T2],
				self.fft_out,
				axes=(-1,), direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',)
			)
			self.fft_bwd = pyfftw.FFTW(
				self.proj_in,
				self.proj[:,:T2],
				axes=(-1,), direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE',), normalise_idft=False
			)
		
		self.nnz = np.zeros(self.n_trials, dtype=np.int32)
		self.nz_index = np.empty((self.n_trials, self.n_times // self.n_times_atom, 2), dtype=np.int32)
		self.nz_coeff = np.empty((self.n_trials, self.n_times // self.n_times_atom), dtype=np.float64)
		self.total_nnz = 0

		self.z_hat = None
		self.nnz_atom = None
		self.z_hat_computed = False

		self.cost = self.XtX

	def compute_z(self):
		self.cost = self.XtX
		self.z_hat_computed = False

		if self.use_fft:
			# TODO: use batches of size 8 instead of size p
			# in order to not allocate too large buffers
			self.fft_data[...,:self.n_times_atom] = self.D_hat[...,::-1]
			self.fft_data[...,self.n_times_atom:] = 0
			self.fft_fwd()
			for trial in range(self.n_trials):
				np.einsum("ict,ct->it", self.fft_out, self.X_fft[trial], out=self.proj_in)
				self.fft_bwd()
				_dp_updt_fft(self.proj, self.n_times_atom, self.reg, self.dp, self.last, self.atom_index, self.atom_coeff)
				self.nnz[trial] = _get_nz_values(self.last, self.atom_index, self.atom_coeff, self.n_times_atom,
											self.nz_index[trial], self.nz_coeff[trial])
				self.cost += self.dp[-1]
		else:
			for trial in range(self.n_trials):
				_dp_updt_prod(self.D_hat, self.X[trial], self.reg, self.dp, self.last, self.atom_index, self.atom_coeff)
				self.nnz[trial] = _get_nz_values(self.last, self.atom_index, self.atom_coeff, self.n_times_atom,
											self.nz_index[trial], self.nz_coeff[trial])
				self.cost += self.dp[-1]
		
		self.total_nnz = self.nnz.sum()

	def compute_objective(self, D):
		return _compute_objective(D, self.X, self.nnz, self.nz_index, self.nz_coeff)

	def get_cost(self):
		return self.cost

	def _compute_dense_z_hat(self):
		if not self.z_hat_computed:
			if self.z_hat is None:
				self.z_hat = np.empty(self.get_z_hat_shape(), dtype=np.float64)
				self.nnz_atom = np.empty(self.n_atoms, dtype=np.int32)
			_compute_z_hat(self.nnz, self.nz_index, self.nz_coeff, self.z_hat)
			self.z_hat_computed = True

	def get_z_hat(self):
		self._compute_dense_z_hat()
		return self.z_hat

	def get_z_nnz(self):
		return self.nnz_atom

	def set_D(self, D):
		self.D_hat = D