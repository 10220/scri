import copy
import numpy as np
from scipy.interpolate import CubicSpline
import spherical_functions
import quaternion
from . import jit, Coprecessing, Coorbital, Corotating, Inertial


class ModesTimeSeries(spherical_functions.Modes):
    """Object to store SWSH modes as functions of time

    This class subclasses the spinsfast.Modes class, but also tracks corresponding time values,
    allowing this class to have extra methods for interpolation, as well as differentiation and
    integration in time.

    NOTE: The time array is not copied; this class merely keeps a reference to the original time
    array.  If you change that array *in place* outside of this class, it changes inside of this
    class as well.  You can, of course, change the variable you used to label that array to point to
    some other quantity without affecting the time array stored in this class.

    """

    def __new__(cls, input_array, *args, **kwargs):
        if len(args) > 2:
            raise ValueError("Only one positional argument may be passed")
        if len(args) == 1:
            kwargs["time"] = args[0]
        metadata = copy.copy(getattr(input_array, "_metadata", {}))
        metadata.update(**kwargs)
        input_array = np.asanyarray(input_array).view(complex)
        time = metadata.get("time", None)
        if time is None:
            raise ValueError("Time data must be specified as part of input array or as constructor parameter")
        time = np.asarray(time).view(float)
        if time.ndim != 1:
            raise ValueError(f"Input time array must have exactly 1 dimension; it has {time.ndim}.")
        if input_array.ndim == 0:
            input_array = input_array[np.newaxis, np.newaxis]
        elif input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        elif input_array.shape[-2] != time.shape[0] and input_array.shape[-2] != 1:
            raise ValueError(
                f"Second-to-last axis of input array must have size 1 or same size as time array.\n            Their shapes are {input_array.shape} and {time.shape}, respectively."
            )
        obj = spherical_functions.Modes(input_array, **kwargs).view(cls)
        obj._metadata["time"] = time
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        if "time" not in self._metadata:
            self._metadata["time"] = None

    @property
    def time(self):
        return self._metadata["time"]

    @time.setter
    def time(self, new_time):
        self._metadata["time"][:] = new_time
        return self.time

    @property
    def n_times(self):
        return self.time.size

    u = time

    t = time

    def interpolate(self, new_time, derivative_order=0, out=None):
        new_time = np.asarray(new_time)
        if new_time.ndim != 1:
            raise ValueError(f"New time array must have exactly 1 dimension; it has {new_time.ndim}.")
        new_shape = self.shape[:-2] + (new_time.size, self.shape[-1])
        if out is not None:
            out = np.asarray(out)
            if out.shape != new_shape:
                raise ValueError(
                    f"Output array should have shape {new_shape} for consistency with new time array and modes array"
                )
            if out.dtype != np.complex:
                raise ValueError(f"Output array should have dtype `complex`; it has dtype {out.dtype}")
        result = out or np.empty(new_shape, dtype=complex)
        if derivative_order > 3:
            raise ValueError(
                f"{type(self)} interpolation uses CubicSpline, and cannot take a derivative of order {derivative_order}"
            )
        spline = CubicSpline(self.u, self.view(np.ndarray), axis=-2)
        if derivative_order < 0:
            spline = spline.antiderivative(-derivative_order)
        elif 0 < derivative_order <= 3:
            spline = spline.derivative(derivative_order)
        result[:] = spline(new_time)
        metadata = self._metadata.copy()
        metadata["time"] = new_time
        return type(self)(result, **metadata)

    def antiderivative(self, antiderivative_order=1):
        """Integrate modes with respect to time"""
        return self.interpolate(self.time, derivative_order=-antiderivative_order)

    def derivative(self, derivative_order=1):
        """Differentiate modes with respect to time"""
        return self.interpolate(self.time, derivative_order=derivative_order)

    @property
    def dot(self):
        """Differentiate modes once with respect to time"""
        return self.derivative()

    @property
    def ddot(self):
        """Differentiate modes twice with respect to time"""
        return self.derivative(2)

    @property
    def int(self):
        """Integrate modes once with respect to time"""
        return self.antiderivative()

    @property
    def iint(self):
        """Integrate modes twice with respect to time"""
        return self.antiderivative(2)

    @property
    def LM(self):
        return np.array([[ell, m] for ell in range(self.ell_min, self.ell_max + 1) for m in range(-ell, ell + 1)])

    @property
    def eth_GHP(self):
        """Raise spin-weight with GHP convention"""
        return self.eth / np.sqrt(2)

    @property
    def ethbar_GHP(self):
        """Lower spin-weight with GHP convention"""
        return self.ethbar / np.sqrt(2)

    def grid_multiply(self, mts, **kwargs):
        """This is a hack for now until the Wigner-D calculator take less time
        and memory to compute. This will compute the values of self and abd on
        a grid, multiply the grid values together, and then return the mode
        coefficients of the product.

        Parameters
        ----------
        self: ModesTimeSeries
            One of the quantities to multiply.
        mts: ModesTimeSeries
            The quantity to multiply with 'self'.

        Keyword Parameters
        ------------------
        working_ell_max: int, optional
            The value of ell_max to be used to define the computation grid. The
            number of theta points and the number of phi points are set to
            2*working_ell_max+1. Defaults to 2*self.ell_max.
        output_ell_max: int, optional
            The value of ell_max in the output mts object. Defaults to self.ell_max.

        """
        import spinsfast
        import spherical_functions as sf
        from spherical_functions import LM_index

        output_ell_max = kwargs.pop("output_ell_max") if "output_ell_max" in kwargs else self.ell_max
        working_ell_max = kwargs.pop("working_ell_max") if "working_ell_max" in kwargs else 2 * self.ell_max
        n_theta = n_phi = 2 * working_ell_max + 1

        if (self.t == mts.t).all():
            n_times = self.n_times

        # Transform to grid representation
        self_grid = np.empty((n_times, n_theta, n_phi), dtype=complex)
        mts_grid = self_grid.copy()
        for t_i in range(self.n_times):
            self_grid[t_i, :, :] = spinsfast.salm2map(
                self.ndarray[t_i, :], self.spin_weight, lmax=self.ell_max, Ntheta=n_theta, Nphi=n_phi
            )
            mts_grid[t_i, :, :] = spinsfast.salm2map(
                mts.ndarray[t_i, :], mts.spin_weight, lmax=mts.ell_max, Ntheta=n_theta, Nphi=n_phi
            )

        product_grid = self_grid * mts_grid
        product_spin_weight = self.spin_weight + mts.spin_weight

        # Transform back to mode representation
        product = np.empty((n_times, (working_ell_max) ** 2), dtype=complex)
        for t_i in range(self.n_times):
            product[t_i, :] = spinsfast.map2salm(product_grid[t_i, :], product_spin_weight, lmax=working_ell_max - 1)

        # Convert product ndarray to a ModesTimeSeries object
        product = product[:, : LM_index(output_ell_max, output_ell_max, 0) + 1]
        product = ModesTimeSeries(
            sf.SWSH_modes.Modes(
                product,
                spin_weight=product_spin_weight,
                ell_min=0,
                ell_max=output_ell_max,
                multiplication_truncator=max,
            ),
            time=self.t,
        )
        return product

    from .mode_calculations_MTS import LLMatrix, LdtVector, angular_velocity

    def rotate_physical_system(self, R_phys):
        """Rotate a Waveform in place

        This just rotates the decomposition basis by the inverse of the input
        rotor(s).  See `rotate_decomposition_basis`.

        For more information on the analytical details, see
        http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

        """
        rotated_data = rotate_decomposition_basis(self, ~R_phys)
        return self  # Probably no return, but just in case...

    def rotate_decomposition_basis(self, R_basis):
        """Rotate a Waveform in place

        This function takes a Waveform object `self` and either a quaternion
        or array of quaternions `R_basis`.  It applies that rotation to
        the decomposition basis of the modes in the Waveform.  The change
        in basis is also recorded in the Waveform's `frame` data.

        For more information on the analytical details, see
        http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

        """
        # This will be used in the jitted functions below to store the
        # Wigner D matrix at each time step
        D = np.empty((spherical_functions.WignerD._total_size_D_matrices(self.ell_min, self.ell_max),), dtype=complex)

        rotated_data = self.copy()

        if isinstance(R_basis, (list, np.ndarray)) and len(R_basis) == 1:
            R_basis = R_basis[0]

        if isinstance(R_basis, (list, np.ndarray)):
            if isinstance(R_basis, np.ndarray) and R_basis.ndim != 1:
                raise ValueError("Input dimension mismatch.  R_basis.shape={1}".format(R_basis.shape))
            if self.shape[0] != len(R_basis):
                raise ValueError(
                    "Input dimension mismatch.  (self.shape[0]={}) != (len(R_basis)={})".format(
                        self.shape[0], len(R_basis)
                    )
                )
            _rotate_decomposition_basis_by_series(
                rotated_data, quaternion.as_spinor_array(R_basis), self.ell_min, self.ell_max, D
            )

        # We can't just use an `else` here because we need to process the
        # case where the input was an iterable of length 1, which we've
        # now changed to just a single quaternion.
        if isinstance(R_basis, np.quaternion):
            spherical_functions._Wigner_D_matrices(R_basis.a, R_basis.b, self.ell_min, self.ell_max, D)
            tmp = np.empty((2 * self.ell_max + 1,), dtype=complex)
            _rotate_decomposition_basis_by_constant(rotated_data, self.ell_min, self.ell_max, D, tmp)

        return rotated_data


@jit("void(c16[:,:], i8, i8, c16[:], c16[:])")
def _rotate_decomposition_basis_by_constant(data, ell_min, ell_max, D, tmp):
    """Rotate data by the same rotor at each point in time

    `D` is the Wigner D matrix for all the ell values.

    `tmp` is just a workspace used as temporary storage to hold the
    results for each item of data during the sum.

    """
    for i_t in range(data.shape[0]):
        for ell in range(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = spherical_functions._linear_matrix_offset(ell, ell_min)

            for i_m in range(2 * ell + 1):
                tmp[i_m] = 0j
            for i_mp in range(2 * ell + 1):
                for i_m in range(2 * ell + 1):
                    tmp[i_m] += data[i_t, i_data + i_mp] * D[i_D + (2 * ell + 1) * i_mp + i_m]
            for i_m in range(2 * ell + 1):
                data[i_t, i_data + i_m] = tmp[i_m]


@jit("void(c16[:,:], c16[:,:], i8, i8, c16[:])")
def _rotate_decomposition_basis_by_series(data, R_basis, ell_min, ell_max, D):
    """Rotate data by a different rotor at each point in time

    `D` is just a workspace, which holds the Wigner D matrices.
    During the summation, it is also used as temporary storage to hold
    the results for each item of data, where the first row in the
    matrix is overwritten with the new sums.

    """
    for i_t in range(data.shape[0]):
        spherical_functions._Wigner_D_matrices(R_basis[i_t, 0], R_basis[i_t, 1], ell_min, ell_max, D)
        for ell in range(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = spherical_functions._linear_matrix_offset(ell, ell_min)

            for i_m in range(2 * ell + 1):
                new_data_mp = 0j
                for i_mp in range(2 * ell + 1):
                    new_data_mp += data[i_t, i_data + i_mp] * D[i_D + i_m + (2 * ell + 1) * i_mp]
                D[i_D + i_m] = new_data_mp
            for i_m in range(2 * ell + 1):
                data[i_t, i_data + i_m] = D[i_D + i_m]
