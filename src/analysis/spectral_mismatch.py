"""Spectral Mismatch Correction Module (IEC 60904-7).

Implements spectral mismatch factor (M) calculation for correcting
PV measurements taken under non-standard spectral conditions.

The spectral mismatch factor accounts for:
- Difference between simulator spectrum and AM1.5G reference
- Difference between test device and reference cell spectral responses
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.iec_standards import (
    WAVELENGTH_BANDS,
    SPECTRAL_MISMATCH_CONFIG,
    SimulatorClass,
    SPECTRAL_MATCH_CRITERIA
)


@dataclass
class SpectralData:
    """Spectral irradiance or response data."""
    wavelengths_nm: np.ndarray  # Wavelength values (nm)
    values: np.ndarray  # Irradiance (W/m²/nm) or SR (A/W)
    name: str = ""
    unit: str = ""

    def __post_init__(self):
        """Validate and sort data."""
        # Convert to numpy arrays
        self.wavelengths_nm = np.array(self.wavelengths_nm, dtype=float)
        self.values = np.array(self.values, dtype=float)

        # Sort by wavelength
        sort_idx = np.argsort(self.wavelengths_nm)
        self.wavelengths_nm = self.wavelengths_nm[sort_idx]
        self.values = self.values[sort_idx]

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Return wavelength range."""
        return (float(self.wavelengths_nm.min()), float(self.wavelengths_nm.max()))

    def interpolate(self, new_wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate to new wavelength grid.

        Args:
            new_wavelengths: Target wavelength array (nm)

        Returns:
            Interpolated values
        """
        f = interp1d(
            self.wavelengths_nm,
            self.values,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        return f(new_wavelengths)

    def integrate_over_range(
        self,
        wl_min: float,
        wl_max: float
    ) -> float:
        """Integrate values over wavelength range.

        Args:
            wl_min: Minimum wavelength (nm)
            wl_max: Maximum wavelength (nm)

        Returns:
            Integrated value
        """
        mask = (self.wavelengths_nm >= wl_min) & (self.wavelengths_nm <= wl_max)
        if np.sum(mask) < 2:
            return 0.0

        return float(integrate.trapezoid(
            self.values[mask],
            self.wavelengths_nm[mask]
        ))


# ============================================================================
# AM1.5G REFERENCE SPECTRUM (IEC 60904-3)
# ============================================================================

def get_am15g_reference() -> SpectralData:
    """Get AM1.5G reference spectrum (simplified).

    Returns standard AM1.5G spectrum for spectral mismatch calculations.
    In production, this would load from a data file with full resolution.

    Returns:
        SpectralData with AM1.5G spectrum
    """
    # Simplified AM1.5G spectrum (key wavelengths and irradiance values)
    # Full spectrum would have 2002 data points from 280-4000nm
    wavelengths = np.array([
        300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200
    ], dtype=float)

    # Spectral irradiance (W/m²/nm) - approximate values
    irradiance = np.array([
        0.02, 0.25, 1.15, 1.75, 1.85, 1.65, 1.55, 1.50, 1.30, 1.15,
        1.00, 0.90, 0.75, 0.60, 0.55, 0.45, 0.35, 0.30, 0.25
    ], dtype=float)

    return SpectralData(
        wavelengths_nm=wavelengths,
        values=irradiance,
        name="AM1.5G",
        unit="W/m²/nm"
    )


# ============================================================================
# TYPICAL SPECTRAL RESPONSES
# ============================================================================

def get_typical_sr_csi() -> SpectralData:
    """Get typical spectral response for crystalline silicon.

    Returns:
        SpectralData with c-Si spectral response
    """
    wavelengths = np.array([
        300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200
    ], dtype=float)

    # Relative spectral response (A/W normalized)
    sr = np.array([
        0.10, 0.35, 0.55, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97,
        0.95, 0.90, 0.80, 0.60, 0.35, 0.15, 0.05, 0.01, 0.00
    ], dtype=float)

    return SpectralData(
        wavelengths_nm=wavelengths,
        values=sr,
        name="c-Si",
        unit="A/W (normalized)"
    )


def get_typical_sr_cdte() -> SpectralData:
    """Get typical spectral response for CdTe.

    Returns:
        SpectralData with CdTe spectral response
    """
    wavelengths = np.array([
        300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        800, 850, 900, 950, 1000
    ], dtype=float)

    sr = np.array([
        0.20, 0.65, 0.85, 0.92, 0.95, 0.93, 0.90, 0.85, 0.75, 0.50,
        0.20, 0.05, 0.01, 0.00, 0.00
    ], dtype=float)

    return SpectralData(
        wavelengths_nm=wavelengths,
        values=sr,
        name="CdTe",
        unit="A/W (normalized)"
    )


# ============================================================================
# SPECTRAL MISMATCH FACTOR CALCULATION
# ============================================================================

def calculate_mismatch_factor(
    simulator_spectrum: SpectralData,
    reference_spectrum: SpectralData,
    test_device_sr: SpectralData,
    reference_cell_sr: SpectralData,
    wl_min: float = 300.0,
    wl_max: float = 1200.0,
    wl_step: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """Calculate spectral mismatch factor M per IEC 60904-7.

    M = [∫E_sim·SR_rc·dλ / ∫E_ref·SR_rc·dλ] × [∫E_ref·SR_td·dλ / ∫E_sim·SR_td·dλ]

    Where:
    - E_sim: Simulator spectral irradiance
    - E_ref: Reference spectrum (AM1.5G)
    - SR_rc: Reference cell spectral response
    - SR_td: Test device spectral response

    Args:
        simulator_spectrum: Measured simulator spectrum
        reference_spectrum: Reference spectrum (typically AM1.5G)
        test_device_sr: Spectral response of test device
        reference_cell_sr: Spectral response of reference cell
        wl_min: Minimum wavelength for integration (nm)
        wl_max: Maximum wavelength for integration (nm)
        wl_step: Wavelength step for interpolation (nm)

    Returns:
        (M_factor, details_dict)
    """
    # Create common wavelength grid
    wavelengths = np.arange(wl_min, wl_max + wl_step, wl_step)

    # Interpolate all spectra to common grid
    e_sim = simulator_spectrum.interpolate(wavelengths)
    e_ref = reference_spectrum.interpolate(wavelengths)
    sr_rc = reference_cell_sr.interpolate(wavelengths)
    sr_td = test_device_sr.interpolate(wavelengths)

    # Calculate integrals
    # Numerator term 1: E_sim × SR_rc
    int_sim_rc = integrate.trapezoid(e_sim * sr_rc, wavelengths)

    # Denominator term 1: E_ref × SR_rc
    int_ref_rc = integrate.trapezoid(e_ref * sr_rc, wavelengths)

    # Numerator term 2: E_ref × SR_td
    int_ref_td = integrate.trapezoid(e_ref * sr_td, wavelengths)

    # Denominator term 2: E_sim × SR_td
    int_sim_td = integrate.trapezoid(e_sim * sr_td, wavelengths)

    # Calculate M factor
    if int_ref_rc == 0 or int_sim_td == 0:
        m_factor = 1.0  # Default if calculation impossible
    else:
        m_factor = (int_sim_rc / int_ref_rc) * (int_ref_td / int_sim_td)

    details = {
        "m_factor": float(m_factor),
        "integral_sim_rc": float(int_sim_rc),
        "integral_ref_rc": float(int_ref_rc),
        "integral_ref_td": float(int_ref_td),
        "integral_sim_td": float(int_sim_td),
        "wavelength_range": (wl_min, wl_max),
        "data_points": len(wavelengths)
    }

    return (float(m_factor), details)


def calculate_mismatch_correction(
    isc_measured: float,
    m_factor: float
) -> float:
    """Apply spectral mismatch correction to Isc.

    Isc_corrected = Isc_measured × M

    Args:
        isc_measured: Measured short-circuit current (A)
        m_factor: Spectral mismatch factor

    Returns:
        Corrected Isc (A)
    """
    return isc_measured * m_factor


# ============================================================================
# SPECTRAL MATCH CLASSIFICATION (IEC 60904-9)
# ============================================================================

def calculate_band_ratios(
    simulator_spectrum: SpectralData,
    reference_spectrum: SpectralData
) -> Dict[str, float]:
    """Calculate spectral match ratios for each wavelength band.

    Per IEC 60904-9, the ratio of simulator to reference irradiance
    is calculated for each of the 6 defined wavelength bands.

    Args:
        simulator_spectrum: Measured simulator spectrum
        reference_spectrum: Reference spectrum (AM1.5G)

    Returns:
        Dictionary mapping band names to ratios
    """
    ratios = {}

    for band in WAVELENGTH_BANDS:
        # Integrate simulator spectrum over band
        sim_integral = simulator_spectrum.integrate_over_range(
            band.wavelength_min_nm,
            band.wavelength_max_nm
        )

        # Integrate reference spectrum over band
        ref_integral = reference_spectrum.integrate_over_range(
            band.wavelength_min_nm,
            band.wavelength_max_nm
        )

        # Calculate ratio
        if ref_integral > 0:
            ratio = sim_integral / ref_integral
        else:
            ratio = 1.0

        ratios[band.name] = float(ratio)

    return ratios


def classify_spectral_match(
    band_ratios: Dict[str, float]
) -> Tuple[SimulatorClass, Dict[str, SimulatorClass]]:
    """Classify spectral match per IEC 60904-9.

    Args:
        band_ratios: Dictionary of band name to ratio

    Returns:
        (overall_class, per_band_classes)
    """
    band_classes = {}
    class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]

    for band_name, ratio in band_ratios.items():
        try:
            band_class = SPECTRAL_MATCH_CRITERIA.classify(ratio)
        except ValueError:
            band_class = SimulatorClass.C
        band_classes[band_name] = band_class

    # Overall class is worst of all bands
    if band_classes:
        worst_idx = max(class_order.index(c) for c in band_classes.values())
        overall_class = class_order[worst_idx]
    else:
        overall_class = SimulatorClass.C

    return (overall_class, band_classes)


# ============================================================================
# UNCERTAINTY IN SPECTRAL MISMATCH
# ============================================================================

@dataclass
class MismatchUncertainty:
    """Uncertainty components for spectral mismatch factor."""
    u_simulator_spectrum: float = 0.5  # % uncertainty in simulator spectrum
    u_reference_spectrum: float = 0.5  # % uncertainty in AM1.5G
    u_test_device_sr: float = 1.0  # % uncertainty in test device SR
    u_reference_cell_sr: float = 0.5  # % uncertainty in reference cell SR
    u_wavelength: float = 0.1  # % uncertainty from wavelength accuracy

    def combined_uncertainty(self) -> float:
        """Calculate combined standard uncertainty for M factor.

        Returns:
            Combined standard uncertainty (%)
        """
        # Root sum of squares for uncorrelated sources
        u_c = np.sqrt(
            self.u_simulator_spectrum ** 2 +
            self.u_reference_spectrum ** 2 +
            self.u_test_device_sr ** 2 +
            self.u_reference_cell_sr ** 2 +
            self.u_wavelength ** 2
        )
        return float(u_c)

    def expanded_uncertainty(self, k: float = 2.0) -> float:
        """Calculate expanded uncertainty.

        Args:
            k: Coverage factor (default 2 for 95% confidence)

        Returns:
            Expanded uncertainty (%)
        """
        return self.combined_uncertainty() * k


def estimate_m_factor_uncertainty(
    m_factor: float,
    uncertainty: MismatchUncertainty
) -> Dict[str, float]:
    """Estimate uncertainty in spectral mismatch factor.

    Args:
        m_factor: Calculated M factor
        uncertainty: Uncertainty components

    Returns:
        Dictionary with uncertainty values
    """
    u_c = uncertainty.combined_uncertainty()
    u_exp = uncertainty.expanded_uncertainty()

    return {
        "m_factor": m_factor,
        "standard_uncertainty_percent": u_c,
        "expanded_uncertainty_percent": u_exp,
        "coverage_factor": 2.0,
        "m_factor_lower": m_factor * (1 - u_exp / 100),
        "m_factor_upper": m_factor * (1 + u_exp / 100)
    }


# ============================================================================
# SIMPLIFIED M FACTOR ESTIMATION
# ============================================================================

def estimate_m_factor_simplified(
    test_device_technology: str,
    reference_cell_technology: str,
    simulator_type: str = "xenon"
) -> Tuple[float, float]:
    """Estimate M factor using typical values (simplified method).

    For cases where full spectral data is not available.

    Args:
        test_device_technology: PV technology (e.g., 'c-Si', 'CdTe', 'CIGS')
        reference_cell_technology: Reference cell technology
        simulator_type: Simulator lamp type ('xenon', 'led', 'halogen')

    Returns:
        (estimated_m_factor, estimated_uncertainty_percent)
    """
    # Typical M factor ranges by technology combination
    m_factor_table = {
        # (test_device, reference_cell, simulator): (M, uncertainty)
        ("c-Si", "c-Si", "xenon"): (1.000, 0.5),
        ("c-Si", "c-Si", "led"): (0.995, 1.0),
        ("c-Si", "c-Si", "halogen"): (0.990, 2.0),

        ("CdTe", "c-Si", "xenon"): (1.005, 1.5),
        ("CdTe", "c-Si", "led"): (1.010, 2.0),

        ("CIGS", "c-Si", "xenon"): (0.998, 1.5),
        ("CIGS", "c-Si", "led"): (1.002, 2.0),

        ("HJT", "c-Si", "xenon"): (1.002, 0.8),
        ("HJT", "c-Si", "led"): (0.998, 1.0),

        ("Perovskite", "c-Si", "xenon"): (1.010, 2.5),
        ("Perovskite", "c-Si", "led"): (1.005, 2.0),
    }

    key = (test_device_technology, reference_cell_technology, simulator_type.lower())

    if key in m_factor_table:
        return m_factor_table[key]
    else:
        # Default values
        return (1.000, 2.0)


# ============================================================================
# SPECTRAL DATA I/O
# ============================================================================

def load_spectrum_from_csv(
    filepath: str,
    wavelength_col: int = 0,
    value_col: int = 1,
    skip_rows: int = 1,
    delimiter: str = ","
) -> SpectralData:
    """Load spectral data from CSV file.

    Args:
        filepath: Path to CSV file
        wavelength_col: Column index for wavelengths
        value_col: Column index for values
        skip_rows: Number of header rows to skip
        delimiter: Column delimiter

    Returns:
        SpectralData object
    """
    data = np.loadtxt(
        filepath,
        delimiter=delimiter,
        skiprows=skip_rows
    )

    return SpectralData(
        wavelengths_nm=data[:, wavelength_col],
        values=data[:, value_col],
        name=Path(filepath).stem
    )


def generate_spectral_report(
    m_factor: float,
    band_ratios: Dict[str, float],
    spectral_class: SimulatorClass,
    uncertainty: Optional[Dict[str, float]] = None
) -> str:
    """Generate spectral mismatch analysis report.

    Args:
        m_factor: Calculated M factor
        band_ratios: Band ratio values
        spectral_class: Overall spectral classification
        uncertainty: Optional uncertainty values

    Returns:
        Formatted report string
    """
    report = """
SPECTRAL MISMATCH ANALYSIS REPORT
==================================

Spectral Mismatch Factor (M): {m_factor:.4f}

Wavelength Band Analysis (IEC 60904-9):
---------------------------------------
""".format(m_factor=m_factor)

    for band in WAVELENGTH_BANDS:
        ratio = band_ratios.get(band.name, 0.0)
        try:
            band_class = SPECTRAL_MATCH_CRITERIA.classify(ratio)
        except ValueError:
            band_class = SimulatorClass.C

        report += f"  {band.name} ({band.range_str} nm): {ratio:.3f} - Class {band_class.value}\n"

    report += f"""
Overall Spectral Classification: Class {spectral_class.value}

"""

    if uncertainty:
        report += f"""Uncertainty Analysis:
---------------------
  Standard Uncertainty: ±{uncertainty.get('standard_uncertainty_percent', 0):.2f}%
  Expanded Uncertainty (k=2): ±{uncertainty.get('expanded_uncertainty_percent', 0):.2f}%
  M Factor Range: {uncertainty.get('m_factor_lower', m_factor):.4f} to {uncertainty.get('m_factor_upper', m_factor):.4f}
"""

    return report
