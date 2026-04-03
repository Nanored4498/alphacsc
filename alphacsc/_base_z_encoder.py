import numpy as np

from .utils.dictionary import get_lambda_max
from .loss_and_gradient import compute_objective

class BaseZEncoder:

    def __init__(self, X, D_hat, n_atoms, n_times_atom, n_jobs,
                 solver_kwargs, reg):

        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.n_jobs = n_jobs

        self.solver_kwargs = solver_kwargs
        self.reg = reg

        self.n_trials, self.n_channels, self.n_times = X.shape
        self.n_times_valid = self.n_times - self.n_times_atom + 1

        self.XtX = np.dot(X.ravel(), X.ravel())

    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        raise NotImplementedError()

    def compute_z_partial(self, i0):
        """
        Compute z on a slice of the signal X, for online learning.

        Parameters
        ----------
        i0 : int
            Slice index.
        """
        raise NotImplementedError()

    def compute_objective(self, D):
        '''Compute the value of the objective function.

        Parameters
        ----------
        D : array, shape (n_atoms, n_channels + n_times_atom) or
                         (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
            D should be feasible.

        Returns
        -------
        obj :
            The value of objective function.
        '''
        return compute_objective(D=D, constants=self.get_constants())

    def get_cost(self):
        """
        Computes the cost of the current sparse representation (z_hat)

        Returns
        -------
        cost: float
            The value of the objective function
        """
        raise NotImplementedError()

    def get_sufficient_statistics(self):
        """
        Computes sufficient statistics to update D.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics.
        """
        raise NotImplementedError()

    def get_sufficient_statistics_partial(self):
        """
        Returns the partial sufficient statistics that were
        computed during the last call to compute_z_partial.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics for the slice that was
            selected in the last call of ``compute_z_partial``
        """
        raise NotImplementedError()

    def get_max_error_patch(self):
        """
        Returns the patch of the signal with the largest reconstuction error.

        Returns
        -------
        D_k : ndarray, shape (n_channels, n_times_atom) or
                (n_channels + n_times_atom,)
            Patch of the residual with the largest error.
        """
        raise NotImplementedError()

    def get_z_hat(self):
        """
        Returns the sparse codes of the signals.

        Returns
        -------
        z_hat : ndarray, shape (n_trials, n_atoms, n_times_valid)
            Sparse codes of the signal X.
        """
        raise NotImplementedError()

    def get_z_hat_shape(self):
        """
        Returns the shape of the sparse codes.

        Returns
        -------
        shape : tuple
            Shape of the sparse code.
        """
        return (self.n_trials, self.n_atoms, self.n_times_valid)

    def get_z_nnz(self):
        """
        Return the number of non-zero activations per atoms for the signals.

        Returns
        -------
        z_nnz : ndarray, shape (n_atoms,)
            Ratio of non-zero activations for each atom.
        """
        raise NotImplementedError()

    def set_D(self, D):
        """
        Update the dictionary.

        Parameters
        ----------
        D : ndarray, shape (n_atoms, n_channels, n_time_atoms)
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
        raise NotImplementedError()

    def update_reg(self, is_per_atom):
        """
        Update the regularization parameter.

        Parameters
        ----------
        is_per_atom: bool
            True if lmbd_max='per_atom'; False otherwise

        """
        self.reg = self.reg * get_lambda_max(self.X, self.D_hat)

        if not is_per_atom:
            self.reg = self.reg.max()

    def get_constants(self):
        """
        """

        return dict(n_channels=self.n_channels, XtX=self.XtX,
                    ztz=self.ztz, ztX=self.ztX)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
