"""
Contains the allowed options for several parameters.
"""

import enum


class IntensityMeasure(enum.StrEnum):
    """
    The intensity measure to use for the hazard curve calculation.

    Valid options are:
    'PGA','SA(0.1)', 'SA(0.15)', 'SA(0.2)', 'SA(0.25)', 'SA(0.3)', 'SA(0.35)', 'SA(0.4)', 'SA(0.5)', 'SA(0.6)',
    'SA(0.7)', 'SA(0.8)', 'SA(0.9)', 'SA(1.0)', 'SA(1.25)', 'SA(1.5)', 'SA(1.75)', 'SA(2.0)', 'SA(2.5)', 'SA(3.0)',
	'SA(3.5)', 'SA(4.0)', 'SA(4.5)', 'SA(5.0)', 'SA(6.0)', 'SA(7.5)', 'SA(10.0)'
    """

    PGA = "PGA"  # Peak ground acceleration
    SA01 = "SA(0.1)"  # Spectral acceleration at 0.1 seconds
    SA015 = "SA(0.15)"  # Spectral acceleration at 0.15 seconds
    SA02 = "SA(0.2)"  # Spectral acceleration at 0.2 seconds
    SA025 = "SA(0.25)"  # Spectral acceleration at 0.25 seconds
    SA03 = "SA(0.3)"  # Spectral acceleration at 0.3 seconds
    SA035 = "SA(0.35)"  # Spectral acceleration at 0.35 seconds
    SA04 = "SA(0.4)"  # Spectral acceleration at 0.4 seconds
    SA05 = "SA(0.5)"  # Spectral acceleration at 0.5 seconds
    SA06 = "SA(0.6)"  # Spectral acceleration at 0.6 seconds
    SA07 = "SA(0.7)"  # Spectral acceleration at 0.7 seconds
    SA08 = "SA(0.8)"  # Spectral acceleration at 0.8 seconds
    SA09 = "SA(0.9)"  # Spectral acceleration at 0.9 seconds
    SA10 = "SA(1.0)"  # Spectral acceleration at 1.0 seconds
    SA125 = "SA(1.25)"  # Spectral acceleration at 1.25 seconds
    SA15 = "SA(1.5)"  # Spectral acceleration at 1.5 seconds
    SA175 = "SA(1.75)"  # Spectral acceleration at 1.75 seconds
    SA20 = "SA(2.0)"  # Spectral acceleration at 2.0 seconds
    SA25 = "SA(2.5)"  # Spectral acceleration at 2.5 seconds
    SA30 = "SA(3.0)"  # Spectral acceleration at 3.0 seconds
    SA35 = "SA(3.5)"  # Spectral acceleration at 3.5 seconds
    SA40 = "SA(4.0)"  # Spectral acceleration at 4.0 seconds
    SA45 = "SA(4.5)"  # Spectral acceleration at 4.5 seconds
    SA50 = "SA(5.0)"  # Spectral acceleration at 5.0 seconds
    SA60 = "SA(6.0)"  # Spectral acceleration at 6.0 seconds
    SA75 = "SA(7.5)"  # Spectral acceleration at 7.5 seconds
    SA100 = "SA(10.0)"  # Spectral acceleration at 10.0 seconds

class LocationCode(enum.StrEnum):
    """
    The locations to generate hazard curves for. Valid options for
    Auckland, Wellington, and Christchurch are AKL, WLG, and CHC, respectively.
    """

    AKL = "AKL"  # Auckland
    WLG = "WLG"  # Wellington
    CHC = "CHC"  # Christchurch


class InterfaceName(enum.StrEnum):
    """
    The options for which subduction interfaces to include in a logic tree.

    Valid options are:
    only_HIK which includes only the Hikurangi–Kermadec subduction zone
    only_PUY which includes only the Puysegur subduction zone.
    HIK_and_PUY which includes the Hikurangi–Kermadec and Puysegur subduction zones.
    """

    only_HIK = "only_HIK"  # Hikurangi–Kermadec subduction interface
    only_PUY = "only_PUY"  # Puysegur subduction interface
    HIK_and_PUY = "HIK_and_PUY"  # Hikurangi–Kermadec and Puysegur subduction interfaces


class TectonicRegionTypeName(enum.StrEnum):
    """
    The options for tectonic region types in a logic tree.

    Valid options are:
    Active_Shallow_Crust
    Subduction_Interface
    Subduction_Intraslab.
    """

    Active_Shallow_Crust = "Active Shallow Crust"
    Subduction_Interface = "Subduction Interface"
    Subduction_Intraslab = "Subduction Intraslab"