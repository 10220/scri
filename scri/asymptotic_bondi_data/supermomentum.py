# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData
### class.  In particular, they assume that the first argument, `self` is an instance of
### AsymptoticBondData.  They should probably not be used outside of that class.

import numpy as np


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
