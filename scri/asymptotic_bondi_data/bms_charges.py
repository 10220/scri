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

        \\Psi = \\psi_2 + \\eth \\eth \\bar{\\sigma} + \\sigma * \\dot{\\bar{\\sigma}}

    Note that the last term is a product between two fields.  If, for example, these both have
    ell_max=8, then their full product would have ell_max=16, meaning that we would go from
    tracking 81 modes to 289.  This shows that deciding how to truncate the output ell is
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
        return self.psi2 + self.sigma.bar.eth_GHP.eth_GHP + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell)
    elif truncate_ell:
        return (
            self.psi2.truncate_ell(truncate_ell)
            + self.sigma.bar.eth_GHP.eth_GHP.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
        )
    else:
        return self.psi2 + self.sigma.bar.eth_GHP.eth_GHP + self.sigma * self.sigma.bar.dot


def angular_momentum_aspect(self, truncate_ell=max):
    """Compute the Bondi mass aspect of the AsymptoticBondiData

    The Bondi angular momentum aspect is given by

        L = \\psi_1 + \\sigma * \\eth \\bar{\\sigma}} + \\frac{1}{2} \\eth{\\sigma * \\bar{\\sigma}}

    Note that the second and third terms are products between two fields. If, for example, both fields have
    ell_max=8, then their full product would have ell_max=16, meaning that we would go from
    tracking 81 modes to 289.  This shows that deciding how to truncate the output ell is
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
        any of the individual Modes objects in the equation for L above -- but not the
        product.

    """
    if callable(truncate_ell):
        return (
            self.psi1
            + self.sigma.multiply(self.sigma.bar.eth_GHP, truncator=truncate_ell)
            + 0.5 * (self.sigma.multiply(self.sigma.bar, truncator=truncate_ell)).eth_GHP
        )
    elif truncate_ell:
        return (
            self.psi1.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.eth_GHP, truncator=lambda tup: truncate_ell)
            + 0.5 * (self.sigma.multiply(self.sigma.bar, truncator=lambda tup: truncate_ell)).eth_GHP
        )
    else:
        return self.psi1 + self.sigma * self.sigma.bar.eth_GHP + 0.5 * (self.sigma * self.sigma.bar).eth_GHP


def bondi_four_momentum(self):
    """Compute the Bondi four-momentum of the AsymptoticBondiData"""
    P_restricted = -self.mass_aspect(1).view(np.ndarray) / sqrt(4 * pi)  # Compute only the parts we need, ell<=1
    four_momentum = np.empty(P_restricted.shape, dtype=float)
    four_momentum[..., 0] = P_restricted[..., 0].real
    four_momentum[..., 1] = (P_restricted[..., 3] - P_restricted[..., 1]).real / sqrt(6)
    four_momentum[..., 2] = (1j * (P_restricted[..., 3] + P_restricted[..., 1])).real / sqrt(6)
    four_momentum[..., 3] = -P_restricted[..., 2].real / sqrt(3)
    return four_momentum


def bondi_angular_momentum(self, output_dimensionless=False):
    """Compute the Bondi angular momentum four-vector of the AsymptoticBondiData

    Parameters
    ----------
    output_dimensionless : bool (default: False)
        If True, then final result is divided by the square of the Bondi rest mass.
        This would be the Bondi spin if the orbital angular momentum were zero.

    Returns
    -------
    numpy.ndarray

    """
    J_restricted = (
        -1j * self.angular_momentum_aspect(1).view(np.ndarray) / sqrt(4 * pi)
    )  # Compute only the parts we need, ell<=1
    angular_momentum = np.empty(J_restricted.shape, dtype=float)
    angular_momentum[..., 0] = J_restricted[..., 0].real
    angular_momentum[..., 1] = (J_restricted[..., 3] - J_restricted[..., 1]).real / sqrt(6)
    angular_momentum[..., 2] = (1j * (J_restricted[..., 3] + J_restricted[..., 1])).real / sqrt(6)
    angular_momentum[..., 3] = -J_restricted[..., 2].real / sqrt(3)
    if output_dimensionless:
        four_momentum = self.bondi_four_momentum()
        rest_mass_sqr = four_momentum[:, 0] ** 2 - np.sum(four_momentum[:, 1:] ** 2, axis=1)
        angular_momentum = self.bondi_angular_momentum()
        return angular_momentum / rest_mass_sqr[:, np.newaxis]
    else:
        return angular_momentum


def bondi_spin(self, t):
    """Computed the spin part of the Bondi angular momentum at a time t. This is done by performing
    a boost so that the asymptotic Bondi data is in the center of momentum frame at time t.

    Parameters
    ----------
    t : float
        Time at which the Bondi spin vector will be computed

    Returns
    -------
    numpy.ndarray

    """
    if not isinstance(t, (int, float)):
        raise TypeError("The Bondi spin vector can only be computed at one value of time.")

    if t not in self.t:
        # Interpolate the ABD to the time t
        new_times = np.sort(np.append(self.t, t))
        t_idx = np.where(new_times == t)[0][0]
        # We avoid copying self to keep the memory footprint low
        four_momentum = self.interpolate(new_times).bondi_four_momentum()[t_idx]
    else:
        t_idx = np.where(self.t == t)[0][0]
        four_momentum = self.bondi_four_momentum()[t_idx]

    velocity = -four_momentum[1:] / np.sqrt(four_momentum[0] ** 2 - np.sum(four_momentum[1:] ** 2))
    boosted_abd = self.transform(time_translation=t, boost_velocity=velocity)
    t_idx = np.where(boosted_abd.t == 0.0)[0][0]
    spin = boosted_abd.bondi_angular_momentum(output_dimensionless=True)[t_idx, 1:]
    return spin


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
        supermomentum = self.psi2 + self.sigma * self.sigma.bar.dot
    elif supermomentum_def.lower() in ["moreschi", "m"]:
        supermomentum = self.psi2 + self.sigma * self.sigma.bar.dot + self.sigma.bar.eth_GHP.eth_GHP
    elif supermomentum_def.lower() in ["geroch", "g"]:
        supermomentum = (
            self.psi2
            + self.sigma * self.sigma.bar.dot
            + 0.5 * (self.sigma.bar.eth_GHP.eth_GHP - self.sigma.ethbar_GHP.ethbar_GHP)
        )
    elif supermomentum_def.lower() in ["geroch-winicour", "gw"]:
        supermomentum = self.psi2 + self.sigma * self.sigma.bar.dot - self.sigma.ethbar_GHP.ethbar_GHP
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
