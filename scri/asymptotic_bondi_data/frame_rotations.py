# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
import quaternion
from .. import jit, Coprecessing, Coorbital, Corotating, Inertial


def to_inertial_frame(self):
    self.rotate_decomposition_basis(~self.frame)
    self.frameType = Inertial
    return self


def rotate_physical_system(self, R_phys):
    """Rotate a Waveform in place

    This just rotates the decomposition basis by the inverse of the input
    rotor(s).  See `rotate_decomposition_basis`.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    self = rotate_decomposition_basis(self, ~R_phys)
    return self  # Probably no return, but just in case...


def rotate_decomposition_basis(self, R_basis):
    """Rotate a Waveform in place

    This function takes a Waveform object `W` and either a quaternion
    or array of quaternions `R_basis`.  It applies that rotation to
    the decomposition basis of the modes in the Waveform.  The change
    in basis is also recorded in the Waveform's `frame` data.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    # Rotate all the waveform data
    self.sigma = self.sigma.rotate_decomposition_basis(R_basis)
    self.psi4 = self.psi4.rotate_decomposition_basis(R_basis)
    self.psi3 = self.psi3.rotate_decomposition_basis(R_basis)
    self.psi2 = self.psi2.rotate_decomposition_basis(R_basis)
    self.psi1 = self.psi1.rotate_decomposition_basis(R_basis)
    self.psi0 = self.psi0.rotate_decomposition_basis(R_basis)

    # Record the frame information
    if isinstance(R_basis, (list, np.ndarray)) and len(R_basis) == 1:
        R_basis = R_basis[0]

    if isinstance(R_basis, (list, np.ndarray)):
        # Update the frame data, using right-multiplication
        if self.frame.size:
            if self.frame.shape[0] == 1:
                # Numpy can't currently multiply one element times an array
                self.frame = np.array([self.frame * R for R in R_basis])
            else:
                self.frame = self.frame * R_basis
        else:
            self.frame = np.copy(R_basis)

    if isinstance(R_basis, np.quaternion):
        # Update the frame data, using right-multiplication
        if self.frame.size:
            self.frame = self.frame * R_basis
        else:
            self.frame = np.array([R_basis])

    return self
