# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData
### class.  In particular, they assume that the first argument, `self` is an instance of
### AsymptoticBondData.  They should probably not be used outside of that class.

import numpy as np
from math import sqrt, pi


def mass_aspect(self, truncate_ell=max):
    """Compute the Bondi mass aspect of the AsymptoticBondiData

    The Bondi mass aspect is given by

        \\Psi = \\psi_2 + \\eth \\eth \bar{\\sigma} + \\sigma * \\dot{\bar{\\sigma}}

    Note that the last term is a product between two fields.  If, for example, these both have
    ell_max=8, then their full product would have ell_max=16, meaning that we would go from
    tracking 77 modes to 289.  This shows that deciding how to truncate the output ell is
    important, which is why this function has the extra argument that it does.

    Parameters
    ==========
    truncate_ell: int, or callable [defaults to `max`]
        Determines how the ell_max value of the output is determined.  If an integer is passed,
        each term in the output is truncated to have at most that ell_max.  (In particular,
        terms that will not be used in the output are simply not computed, without incurring any
        errors due to aliasing.)  If a callable is passed, it is passed on to the
        spherical_functions.Modes.multiply method.  See that function's docstring for details.
        The default behavior will result in the output having ell_max equal to the largest of
        any of the individual Modes objects in the equation for \\Psi above -- but not the
        product.

    """
    if callable(truncate_ell):
        return self.psi2 + self.sigma.bar.eth.eth + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell)
    elif truncate_ell:
        return (
            self.psi2.truncate_ell(truncate_ell)
            + self.sigma.bar.eth.eth.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
        )
    else:
        return self.psi2 + self.sigma.bar.eth.eth + self.sigma * self.sigma.bar.dot


def bondi_four_momentum(self):
    """Compute the Bondi four-momentum of the AsymptoticBondiData"""
    import spherical_functions as sf

    P_restricted = -self.mass_aspect(1).view(np.ndarray) / sqrt(4 * pi)  # Compute only the parts we need, ell<=1
    four_momentum = np.empty(P_restricted.shape, dtype=float)
    four_momentum[..., 0] = P_restricted[..., 0].real
    four_momentum[..., 1] = (P_restricted[..., 3] - P_restricted[..., 1]).real / sqrt(6)
    four_momentum[..., 2] = (1j * (P_restricted[..., 3] + P_restricted[..., 1])).real / sqrt(6)
    four_momentum[..., 3] = -P_restricted[..., 2].real / sqrt(3)
    return four_momentum


def bondi_angular_momentum(self):
    """Compute the Bondi angular momentum vector via Eq. (8) in T. Dray (1985) [DOI:10.1088/0264-9381/2/1/002]"""
    from spherical_functions import LM_index

    Q = (
        self.psi1
        + self.sigma.grid_multiply(self.sigma.bar.eth_GHP)
        + 0.5 * (self.sigma.grid_multiply(self.sigma.bar)).eth_GHP
    ).ndarray
    angular_momentum = np.real(
        1j
        / np.sqrt(24 * np.pi)
        * np.array(
            [
                Q[:, LM_index(-1, 1, 0)] - Q[:, LM_index(1, 1, 0)],
                -1j * (Q[:, LM_index(-1, 1, 0)] + Q[:, LM_index(1, 1, 0)]),
                np.sqrt(2) * Q[:, LM_index(1, 0, 0)],
            ]
        )
    )
    return angular_momentum


def bondi_spin(self):
    """Computes the Bondi spin angular momentum vector ASSUMING that the orbital part of the angular momentum
    computed from the Bondi data is zero."""
    from spherical_functions import LM_index as lm

    four_momentum = self.bondi_four_momentum()
    rest_mass = four_momentum[-1, 0] ** 2 - np.sum(four_momentum[-1, 1:] ** 2)
    angular_momentum = self.bondi_angular_momentum()
    return (angular_momentum / rest_mass).T


def supermomentum(self, supermomentum_def, integrated=False):
    """Computes the supermomentum of the asymptotic Bondi data. Allows for several different definitions
    of the supermomentum. These differences only apply to ell > 1 modes, so they do not affect the Bondi
    four-momentum. See Eqs. (7-9) in arXiv:1404.2475 for the different supermomentum definitions and links
    to further references.

    Parameters
    ----------
    supermomentum_def : str
        The definition of the supermomentum to be computed. One of the following options (case insensitive)
        can be specified:
          * 'Bondi-Sachs' or 'BS'
          * 'Moreschi' or 'M'
          * 'Geroch' or 'G'
          * 'Geroch-Winicour' or 'GW'
    integrated : bool, default: False
        If true, then return the integrated form of the supermomentum. See Eq. (5) in arXiv:1404.2475.

    Returns
    -------
    ModesTimeSeries

    """
    if supermomentum_def.lower() in ["bondi-sachs", "bs"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot)
    elif supermomentum_def.lower() in ["moreschi", "m"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot) + self.sigma.bar.eth_GHP.eth_GHP
    elif supermomentum_def.lower() in ["geroch", "g"]:
        supermomentum = (
            self.psi2
            + self.sigma.grid_multiply(self.sigma.bar.dot)
            + 0.5 * (self.sigma.bar.eth_GHP.eth_GHP - self.sigma.ethbar_GHP.ethbar_GHP)
        )
    elif supermomentum_def.lower() in ["geroch-winicour", "gw"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot) - self.sigma.ethbar_GHP.ethbar_GHP
    else:
        raise ValueError(
            f"Supermomentum defintion '{supermomentum_def}' not recognized. Please choose one of "
            "the following options:\n"
            "  * 'Bondi-Sachs' or 'BS'\n"
            "  * 'Moreschi' or 'M'\n"
            "  * 'Geroch' or 'G'\n"
            "  * 'Geroch-Winicour' or 'GW'"
        )
    if integrated:
        return -0.5 * supermomentum.bar / np.sqrt(np.pi)
    else:
        return supermomentum


def transform_moreschi_supermomentum(supermomentum, **kwargs):
    """Apply a BMS transformation to the Moreschi supermomentum using the Moreschi formula,
    Eq. (9) of arXiv:gr-qc/0203075. NOTE: This transformation only holds for the Moreschi
    supermomentum!

    It is important to note that the input transformation parameters are applied in this order:

      1. (Super)Translations
      2. Rotation (about the origin)
      3. Boost (about the origin)

    All input parameters refer to the transformation required to take the input data's inertial
    frame onto the inertial frame of the output data's inertial observers.  In what follows, the
    coordinates of and functions in the input inertial frame will be unprimed, while corresponding
    values of the output inertial frame will be primed.

    The translations (space, time, spacetime, or super) can be given in various ways, which may
    override each other.  Ultimately, however, they are essentially combined into a single function
    `α`, representing the supertranslation, which transforms the asymptotic time variable `u` as

        u'(u, θ, ϕ) = u(u, θ, ϕ) - α(θ, ϕ)

    A simple time translation by δt would correspond to

        α(θ, ϕ) = δt  # Independent of (θ, ϕ)

    A pure spatial translation δx would correspond to

        α(θ, ϕ) = -δx · n̂(θ, ϕ)

    where `·` is the usual dot product, and `n̂` is the unit vector in the given direction.


    Parameters
    ----------
    supermomentum: ModesTimeSeries
        The object storing the modes of the original data, which will be transformed in this
        function.  This is the only required argument to this function.
    time_translation: float, optional
        Defaults to zero.  Nonzero overrides corresponding components of `spacetime_translation` and
        `supertranslation` parameters.  Note that this is the actual change in the coordinate value,
        rather than the corresponding mode weight (which is what `supertranslation` represents).
    space_translation : float array of length 3, optional
        Defaults to empty (no translation).  Non-empty overrides corresponding components of
        `spacetime_translation` and `supertranslation` parameters.  Note that this is the actual
        change in the coordinate value, rather than the corresponding mode weight (which is what
        `supertranslation` represents).
    spacetime_translation : float array of length 4, optional
        Defaults to empty (no translation).  Non-empty overrides corresponding components of
        `supertranslation`.  Note that this is the actual change in the coordinate value, rather
        than the corresponding mode weight (which is what `supertranslation` represents).
    supertranslation : complex array [defaults to 0]
        This gives the complex components of the spherical-harmonic expansion of the
        supertranslation in standard form, starting from ell=0 up to some ell_max, which may be
        different from the ell_max of the input `supermomentum`. Supertranslations must be real, so
        these values should obey the condition
            α^{ℓ,m} = (-1)^m ᾱ^{ℓ,-m}
        This condition is actually imposed on the input data, so imaginary parts of α(θ, ϕ) will
        essentially be discarded.  Defaults to empty, which causes no supertranslation.  Note that
        some components may be overridden by the parameters above.
    frame_rotation : quaternion [defaults to 1]
        Transformation applied to (x,y,z) basis of the input mode's inertial frame.  For example,
        the basis z vector of the new frame may be written as
           z' = frame_rotation * z * frame_rotation.inverse()
        Defaults to 1, corresponding to the identity transformation (no rotation).
    boost_velocity : float array of length 3 [defaults to (0, 0, 0)]
        This is the three-velocity vector of the new frame relative to the input frame.  The norm of
        this vector is required to be smaller than 1.
    output_ell_max: int [defaults to supermomentum.ell_max]
        Maximum ell value in the output data.
    working_ell_max: int [defaults to 2 * supermomentum.ell_max]
        Maximum ell value to use during the intermediate calculations.  Rotations and time
        translations do not require this to be any larger than supermomentum.ell_max, but other
        transformations will require more values of ell for accurate results.  In particular, boosts
        are multiplied by time, meaning that a large boost of data with large values of time will
        lead to very large power in higher modes.  Similarly, large (super)translations will couple
        power through a lot of modes.  To avoid aliasing, this value should be large, to accomodate
        power in higher modes.

    Returns
    -------
    ModesTimeSeries

    """
    from quaternion import rotate_vectors
    from scipy.interpolate import CubicSpline
    import spherical_functions as sf
    import spinsfast
    import math
    from .transformations import _process_transformation_kwargs, boosted_grid, conformal_factors
    from ..modes_time_series import ModesTimeSeries

    # Parse the input arguments, and define the basic parameters for this function
    frame_rotation, boost_velocity, supertranslation, working_ell_max, output_ell_max, = _process_transformation_kwargs(
        supermomentum.ell_max, **kwargs
    )
    n_theta = 2 * working_ell_max + 1
    n_phi = n_theta
    β = np.linalg.norm(boost_velocity)
    γ = 1 / math.sqrt(1 - β ** 2)

    # Make this into a Modes object, so it can keep track of its spin weight, etc., through the
    # various operations needed below.
    supertranslation = sf.Modes(supertranslation, spin_weight=0).real

    # This is a 2-d array of unit quaternions, which are what the spin-weighted functions should be
    # evaluated on (even for spin 0 functions, for simplicity).  That will be equivalent to
    # evaluating the spin-weighted functions with respect to the transformed grid -- although on the
    # original time slices.
    distorted_grid_rotors = boosted_grid(frame_rotation, boost_velocity, n_theta, n_phi)

    # Compute u, α, Δα, k, ðk/k, 1/k, and 1/k³ on the distorted grid, including new axes to
    # enable broadcasting with time-dependent functions.  Note that the first axis should represent
    # variation in u, the second axis variation in θ', and the third axis variation in ϕ'.
    u = supermomentum.u
    α = sf.Grid(supertranslation.evaluate(distorted_grid_rotors), spin_weight=0).real[np.newaxis, :, :]
    # The factor of 0.25 comes from using the GHP eth instead of the NP eth.
    Δα = sf.Grid(0.25 * supertranslation.ethbar.ethbar.eth.eth.evaluate(distorted_grid_rotors), spin_weight=α.s)[
        np.newaxis, :, :
    ]
    k, ðk_over_k, one_over_k, one_over_k_cubed = conformal_factors(boost_velocity, distorted_grid_rotors)

    # Ψ(u, θ', ϕ') exp(2iλ)
    Ψ = sf.Grid(supermomentum.evaluate(distorted_grid_rotors), spin_weight=0)

    ### The following calculations are done using in-place Horner form.  I suspect this will be the
    ### most efficient form of this calculation, within reason.  Note that the factors of exp(isλ)
    ### were computed automatically by evaluating in terms of quaternions.
    #
    # Ψ'(u, θ', ϕ') = k⁻³ (Ψ - ð²ðbar²α)
    Ψprime_of_timenaught_directionprime = Ψ.copy() - Δα
    Ψprime_of_timenaught_directionprime *= one_over_k_cubed

    # Determine the new time slices.  The set timeprime is chosen so that on each slice of constant
    # u'_i, the average value of u=(u'/k)+α is precisely <u>=u'γ+<α>=u_i.  But then, we have to
    # narrow that set down, so that every grid point on all the u'_i' slices correspond to data in
    # the range of input data.
    timeprime = (u - sf.constant_from_ell_0_mode(supertranslation[0]).real) / γ
    timeprime_of_initialtime_directionprime = k * (u[0] - α)
    timeprime_of_finaltime_directionprime = k * (u[-1] - α)
    earliest_complete_timeprime = np.max(timeprime_of_initialtime_directionprime.view(np.ndarray))
    latest_complete_timeprime = np.min(timeprime_of_finaltime_directionprime.view(np.ndarray))
    timeprime = timeprime[(timeprime >= earliest_complete_timeprime) & (timeprime <= latest_complete_timeprime)]

    # This will store the values of Ψ'(u', θ', ϕ')
    Ψprime_of_timeprime_directionprime = np.zeros((timeprime.size, n_theta, n_phi), dtype=complex)

    # Interpolate the various transformed function values on the transformed grid from the original
    # time coordinate to the new set of time coordinates, independently for each direction.
    for i in range(n_theta):
        for j in range(n_phi):
            k_i_j = k[0, i, j]
            α_i_j = α[0, i, j]
            # u'(u, θ', ϕ')
            timeprime_of_timenaught_directionprime_i_j = k_i_j * (u - α_i_j)
            # Ψ'(u', θ', ϕ')
            Ψprime_of_timeprime_directionprime[:, i, j] = CubicSpline(
                timeprime_of_timenaught_directionprime_i_j, Ψprime_of_timenaught_directionprime[:, i, j], axis=0
            )(timeprime)

    # Finally, transform back from the distorted grid to the SWSH mode weights as measured in that
    # grid.  I'll abuse notation slightly here by indicating those "distorted" mode weights with
    # primes, so that Ψ'(u')_{ℓ', m'} = ∫ Ψ'(u', θ', ϕ') sȲ_{ℓ', m'}(θ', ϕ') sin(θ') dθ' dϕ'
    supermomentum_prime = spinsfast.map2salm(Ψprime_of_timeprime_directionprime, 0, output_ell_max)
    supermomentum_prime = ModesTimeSeries(
        sf.SWSH_modes.Modes(
            supermomentum_prime, spin_weight=0, ell_min=0, ell_max=output_ell_max, multiplication_truncator=max
        ),
        time=timeprime,
    )

    return supermomentum_prime
