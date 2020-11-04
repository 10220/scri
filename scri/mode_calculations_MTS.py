# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from math import sqrt

import numpy as np
import quaternion

from spherical_functions import ladder_operator_coefficient as ladder
from spherical_functions import clebsch_gordan as CG
from . import jit


@jit("void(c16[:,:], c16[:,:], i8[:,:], f8[:,:])")
def _LdtVector(data, datadot, lm, Ldt):
    """Helper function for the LdtVector function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2

    for i_mode in range(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in range(data.shape[0]):
            # Compute first in (+,-,z) basis
            Lp = (
                np.conjugate(data[i_time, i_mode + 1]) * datadot[i_time, i_mode] * ladder(L, M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            Lm = (
                np.conjugate(data[i_time, i_mode - 1]) * datadot[i_time, i_mode] * ladder(L, -M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            Lz = np.conjugate(data[i_time, i_mode]) * datadot[i_time, i_mode] * M

            # Convert into (x,y,z) basis
            Ldt[i_time, 0] += 0.5 * (Lp.imag + Lm.imag)
            Ldt[i_time, 1] += -0.5 * (Lp.real - Lm.real)
            Ldt[i_time, 2] += Lz.imag
    return


def LdtVector(W):
    r"""Calculate the <Ldt> quantity with respect to the modes

    The vector is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    <Ldt>^{a} = \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a | \ell,m > (df/dt)^{\ell,m}

    """
    Ldt = np.zeros((W.shape[0], 3), dtype=float)
    _LdtVector(W.ndarray, W.dot.ndarray, W.LM, Ldt)
    return Ldt


@jit("void(c16[:,:], i8[:,:], f8[:,:,:])")
def _LLMatrix(data, lm, LL):
    """Helper function for the LLMatrix function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2     Im(Lx) =  ( Im(L+) + Im(L-) ) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2     Im(Ly) = -( Re(L+) - Re(L-) ) / 2
    # Lz = Lz             Lz = Lz                   Im(Lz) = Im(Lz)
    # LxLx =   (L+ + L-)(L+ + L-) / 4
    # LxLy = -i(L+ + L-)(L+ - L-) / 4
    # LxLz =   (L+ + L-)(  Lz   ) / 2
    # LyLx = -i(L+ - L-)(L+ + L-) / 4
    # LyLy =  -(L+ - L-)(L+ - L-) / 4
    # LyLz = -i(L+ - L-)(  Lz   ) / 2
    # LzLx =   (  Lz   )(L+ + L-) / 2
    # LzLy = -i(  Lz   )(L+ - L-) / 2
    # LzLz =   (  Lz   )(  Lz   )

    for i_mode in range(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in range(data.shape[0]):
            # Compute first in (+,-,z) basis
            LpLp = (
                np.conjugate(data[i_time, i_mode + 2]) * data[i_time, i_mode] * (ladder(L, M + 1) * ladder(L, M))
                if M + 2 <= L
                else 0.0 + 0.0j
            )
            LpLm = (
                np.conjugate(data[i_time, i_mode]) * data[i_time, i_mode] * (ladder(L, M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LmLp = (
                np.conjugate(data[i_time, i_mode]) * data[i_time, i_mode] * (ladder(L, -(M + 1)) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLm = (
                np.conjugate(data[i_time, i_mode - 2]) * data[i_time, i_mode] * (ladder(L, -(M - 1)) * ladder(L, -M))
                if M - 2 >= -L
                else 0.0 + 0.0j
            )
            LpLz = (
                np.conjugate(data[i_time, i_mode + 1]) * data[i_time, i_mode] * (ladder(L, M) * M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LzLp = (
                np.conjugate(data[i_time, i_mode + 1]) * data[i_time, i_mode] * ((M + 1) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLz = (
                np.conjugate(data[i_time, i_mode - 1]) * data[i_time, i_mode] * (ladder(L, -M) * M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLm = (
                np.conjugate(data[i_time, i_mode - 1]) * data[i_time, i_mode] * ((M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLz = np.conjugate(data[i_time, i_mode]) * data[i_time, i_mode] * M ** 2

            # Convert into (x,y,z) basis
            LxLx = 0.25 * (LpLp + LmLm + LmLp + LpLm)
            LxLy = -0.25j * (LpLp - LmLm + LmLp - LpLm)
            LxLz = 0.5 * (LpLz + LmLz)
            LyLx = -0.25j * (LpLp - LmLp + LpLm - LmLm)
            LyLy = -0.25 * (LpLp - LmLp - LpLm + LmLm)
            LyLz = -0.5j * (LpLz - LmLz)
            LzLx = 0.5 * (LzLp + LzLm)
            LzLy = -0.5j * (LzLp - LzLm)
            # LzLz = (LzLz)

            # Symmetrize
            LL[i_time, 0, 0] += LxLx.real
            LL[i_time, 0, 1] += (LxLy + LyLx).real / 2.0
            LL[i_time, 0, 2] += (LxLz + LzLx).real / 2.0
            LL[i_time, 1, 0] += (LyLx + LxLy).real / 2.0
            LL[i_time, 1, 1] += LyLy.real
            LL[i_time, 1, 2] += (LyLz + LzLy).real / 2.0
            LL[i_time, 2, 0] += (LzLx + LxLz).real / 2.0
            LL[i_time, 2, 1] += (LzLy + LyLz).real / 2.0
            LL[i_time, 2, 2] += LzLz.real
    return


def LLMatrix(W):
    r"""Calculate the <LL> quantity with respect to the modes

    The matrix is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    This quantity is as prescribed by O'Shaughnessy et al.
    <http://arxiv.org/abs/1109.5224>, except that no normalization is
    imposed, and this operates on whatever type of data is input.

    <LL>^{ab} = Re \{ \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a L_b | \ell,m > f^{\ell,m} \}

    """
    LL = np.zeros((W.shape[0], 3, 3), dtype=float)
    _LLMatrix(W.ndarray, W.LM, LL)
    return LL


@jit("void(f8[:,:], f8[:], i8)")
def _LLDominantEigenvector(dpa, dpa_i, i_index):
    """Jitted helper function for LLDominantEigenvector"""
    # Make the initial direction closer to RoughInitialEllDirection than not
    if (dpa_i[0] * dpa[i_index, 0] + dpa_i[1] * dpa[i_index, 1] + dpa_i[2] * dpa[i_index, 2]) < 0.0:
        dpa[i_index, 0] *= -1
        dpa[i_index, 1] *= -1
        dpa[i_index, 2] *= -1
    # Now, go through and make the vectors reasonably continuous.
    d = -1
    LastNorm = sqrt(dpa[i_index, 0] ** 2 + dpa[i_index, 1] ** 2 + dpa[i_index, 2] ** 2)
    for i in range(i_index - 1, -1, -1):
        Norm = dpa[i, 0] ** 2 + dpa[i, 1] ** 2 + dpa[i, 2] ** 2
        dNorm = (dpa[i, 0] - dpa[i - d, 0]) ** 2 + (dpa[i, 1] - dpa[i - d, 1]) ** 2 + (dpa[i, 2] - dpa[i - d, 2]) ** 2
        if dNorm > Norm:
            dpa[i, 0] *= -1
            dpa[i, 1] *= -1
            dpa[i, 2] *= -1
        # While we're here, let's just normalize that last one
        if LastNorm != 0.0 and LastNorm != 1.0:
            dpa[i - d, 0] /= LastNorm
            dpa[i - d, 1] /= LastNorm
            dpa[i - d, 2] /= LastNorm
        LastNorm = sqrt(Norm)
    if LastNorm != 0.0 and LastNorm != 1.0:
        dpa[0, 0] /= LastNorm
        dpa[0, 1] /= LastNorm
        dpa[0, 2] /= LastNorm
    d = 1
    LastNorm = sqrt(dpa[i_index, 0] ** 2 + dpa[i_index, 1] ** 2 + dpa[i_index, 2] ** 2)
    for i in range(i_index + 1, dpa.shape[0]):
        Norm = dpa[i, 0] ** 2 + dpa[i, 1] ** 2 + dpa[i, 2] ** 2
        dNorm = (dpa[i, 0] - dpa[i - d, 0]) ** 2 + (dpa[i, 1] - dpa[i - d, 1]) ** 2 + (dpa[i, 2] - dpa[i - d, 2]) ** 2
        if dNorm > Norm:
            dpa[i, 0] *= -1
            dpa[i, 1] *= -1
            dpa[i, 2] *= -1
        # While we're here, let's just normalize that last one
        if LastNorm != 0.0 and LastNorm != 1.0:
            dpa[i - d, 0] /= LastNorm
            dpa[i - d, 1] /= LastNorm
            dpa[i - d, 2] /= LastNorm
        LastNorm = sqrt(Norm)
    if LastNorm != 0.0 and LastNorm != 1.0:
        dpa[-1, 0] /= LastNorm
        dpa[-1, 1] /= LastNorm
        dpa[-1, 2] /= LastNorm
    return


def LLDominantEigenvector(W, RoughDirection=np.array([0.0, 0.0, 1.0]), RoughDirectionIndex=0):
    r"""Calculate the principal axis of the LL matrix

    \param Lmodes L modes to evaluate (optional)
    \param RoughDirection Vague guess about the preferred initial (optional)

    If Lmodes is empty (default), all L modes are used.  Setting
    Lmodes to [2] or [2,3,4], for example, restricts the range of
    the sum.

    Ell is the direction of the angular velocity for a PN system, so
    some rough guess about that direction allows us to choose the
    direction of the eigenvectors output by this function to be more
    parallel than anti-parallel to that direction.  The default is
    to simply choose the z axis, since this is most often the
    correct choice anyway.

    The vector is given in the (possibly rotating) mode frame
    (X,Y,Z), rather than the inertial frame (x,y,z).

    """
    # Calculate the LL matrix at each instant
    LL = np.zeros((W.shape[0], 3, 3), dtype=float)
    _LLMatrix(W.ndarray, W.LM, LL)

    # This is the eigensystem
    eigenvals, eigenvecs = np.linalg.eigh(LL)

    # Now we find and normalize the dominant principal axis at each
    # moment, made continuous
    dpa = eigenvecs[:, :, 2]  # `eigh` always returns eigenvals in *increasing* order
    _LLDominantEigenvector(dpa, RoughDirection, RoughDirectionIndex)

    return dpa


# @jit
def angular_velocity(self):
    """Angular velocity of Waveform

    This function calculates the angular velocity of a Waveform object from
    its modes.  This was introduced in Sec. II of "Angular velocity of
    gravitational radiation and the corotating frame"
    <http://arxiv.org/abs/1302.2919>.  Essentially, this is the angular
    velocity of the rotating frame in which the time dependence of the modes
    is minimized.

    The vector is given in the (possibly rotating) mode frame (X,Y,Z),
    which is not necessarily equal to the inertial frame (x,y,z).

    """

    # Calculate the <Ldt> vector and <LL> matrix at each instant
    l = self.LdtVector()
    ll = self.LLMatrix()

    # Solve <Ldt> = - <LL> . omega
    omega = -np.linalg.solve(ll, l)

    return omega


def corotating_frame(self, times, R0=quaternion.one, tolerance=1e-12, z_alignment_region=None, return_omega=False):
    """Return rotor taking current mode frame into corotating frame

    This function simply evaluates the angular velocity of the waveform, and
    then integrates it to find the corotating frame itself.  This frame is
    defined to be the frame in which the time-dependence of the waveform is
    minimized --- at least, to the extent possible with a time-dependent
    rotation.  This frame is only unique up to a single overall rotation, which
    is passed in as an optional argument to this function.

    Parameters
    ----------
    self: Waveform
    R0: quaternion [defaults to 1]
        Value of the output rotation at the first output instant
    tolerance: float [defaults to 1e-12]
        Absolute tolerance used in integration
    z_alignment_region: None or 2-tuple of floats [defaults to None]
        If not None, the dominant eigenvector of the <LL> matrix is aligned with the z axis,
        averaging over this portion of the data.  The first and second elements of the input are
        considered fractions of the inspiral at which to begin and end the average.  For example,
        (0.1, 0.9) would lead to starting 10% of the time from the first time step to the max norm
        time, and ending at 90% of that time.
    return_omega: bool [defaults to False]
        If True, return a 2-tuple consisting of the frame (the usual returned object) and the
        angular-velocity data.  That is frequently also needed, so this is just a more efficient
        way of getting the data.

    """
    from quaternion.quaternion_time_series import integrate_angular_velocity, squad

    omega = self.angular_velocity()
    t, frame = integrate_angular_velocity((times, omega), t0=times[0], t1=times[-1], R0=R0, tolerance=tolerance)
    if z_alignment_region is None:
        correction_rotor = quaternion.one
    else:
        initial_time = times[0]
        max_norm_index = self.norm()[slice(self.shape[0] // 4, None, None)].argmax() + (self.shape[0] // 4)
        inspiral_time = times[max_norm_index] - initial_time
        t1 = initial_time + z_alignment_region[0] * inspiral_time
        t2 = initial_time + z_alignment_region[1] * inspiral_time
        i1 = np.argmin(np.abs(times - t1))
        i2 = np.argmin(np.abs(times - t2))
        R = frame[i1:i2]
        i1m = max(0, i1 - 10)
        i1p = i1m + 21
        RoughDirection = omega[i1m + 10]
        Vhat = self[i1:i2].LLDominantEigenvector(RoughDirection=RoughDirection, RoughDirectionIndex=0)
        Vhat_corot = np.array([(Ri.conjugate() * quaternion.quaternion(*Vhati) * Ri).vec for Ri, Vhati in zip(R, Vhat)])
        Vhat_corot_mean = quaternion.quaternion(*np.mean(Vhat_corot, axis=0)).normalized()
        correction_rotor = np.sqrt(-quaternion.z * Vhat_corot_mean).inverse()
    frame = frame * correction_rotor
    frame = frame / np.abs(frame)
    if return_omega:
        return (frame, omega)
    else:
        return frame
