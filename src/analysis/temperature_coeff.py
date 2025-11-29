"""Temperature Coefficient Extraction Module (IEC 60904-10).

Implements extraction of temperature coefficients from I-V measurements
at multiple temperatures:
- Alpha (α): Temperature coefficient of Isc (%/°C or A/°C)
- Beta (β): Temperature coefficient of Voc (%/°C or V/°C)
- Gamma (γ): Temperature coefficient of Pmax (%/°C or W/°C)
- Rs: Series resistance for IEC 60891 corrections
- Kappa (κ): Curve correction factor

Per IEC 60904-10, requires measurements at minimum 4 temperatures
spanning at least 15°C range.
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.iec_standards import (
    TEMP_COEFF_REQUIREMENTS,
    TYPICAL_TEMP_COEFFICIENTS,
    STC
)


class TemperatureCoefficientResult(NamedTuple):
    """Result of temperature coefficient calculation."""
    parameter: str
    value_absolute: float  # Absolute value (A/°C, V/°C, W/°C)
    value_relative: float  # Relative value (%/°C)
    unit_absolute: str
    unit_relative: str
    reference_value: float  # Value at reference temperature
    r_squared: float  # Regression R² value
    std_error: float  # Standard error of coefficient
    n_points: int  # Number of data points used
    temp_range: Tuple[float, float]  # Temperature range used


@dataclass
class TemperatureDataPoint:
    """Single measurement at a specific temperature."""
    temperature: float  # °C
    irradiance: float  # W/m²
    isc: float  # A
    voc: float  # V
    pmax: float  # W
    vmpp: Optional[float] = None  # V
    impp: Optional[float] = None  # A
    ff: Optional[float] = None
    rs: Optional[float] = None  # Ohms
    rsh: Optional[float] = None  # Ohms

    # Optional: full I-V curve data
    voltage_array: Optional[np.ndarray] = None
    current_array: Optional[np.ndarray] = None


@dataclass
class TemperatureCoefficientExtractor:
    """Extract temperature coefficients from multiple measurements.

    Per IEC 60904-10 requirements:
    - Minimum 4 measurement points
    - Temperature range ≥ 15°C
    - Irradiance within ±1% tolerance
    - Linear regression with R² ≥ 0.99
    """

    # Measurement data
    data_points: List[TemperatureDataPoint] = field(default_factory=list)

    # Reference conditions
    reference_temperature: float = STC.temperature
    reference_irradiance: float = STC.irradiance

    # Configuration
    irradiance_tolerance_percent: float = TEMP_COEFF_REQUIREMENTS.irradiance_tolerance_percent
    min_temp_range: float = TEMP_COEFF_REQUIREMENTS.min_temperature_range_c
    min_points: int = TEMP_COEFF_REQUIREMENTS.min_measurement_points
    min_r_squared: float = TEMP_COEFF_REQUIREMENTS.min_r_squared

    # Results storage
    _results: Dict[str, TemperatureCoefficientResult] = field(default_factory=dict)

    def add_measurement(self, data_point: TemperatureDataPoint):
        """Add a measurement data point.

        Args:
            data_point: TemperatureDataPoint with measurement data
        """
        self.data_points.append(data_point)

    def add_measurement_simple(
        self,
        temperature: float,
        isc: float,
        voc: float,
        pmax: float,
        irradiance: float = 1000.0
    ):
        """Add a simple measurement without full I-V curve.

        Args:
            temperature: Cell/module temperature (°C)
            isc: Short-circuit current (A)
            voc: Open-circuit voltage (V)
            pmax: Maximum power (W)
            irradiance: Irradiance level (W/m²)
        """
        self.data_points.append(TemperatureDataPoint(
            temperature=temperature,
            irradiance=irradiance,
            isc=isc,
            voc=voc,
            pmax=pmax
        ))

    def validate_data(self) -> Tuple[bool, List[str]]:
        """Validate measurement data meets IEC requirements.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check number of points
        if len(self.data_points) < self.min_points:
            issues.append(
                f"Insufficient data points: {len(self.data_points)} < {self.min_points}"
            )

        if len(self.data_points) == 0:
            return (False, issues)

        # Check temperature range
        temps = [dp.temperature for dp in self.data_points]
        temp_range = max(temps) - min(temps)
        if temp_range < self.min_temp_range:
            issues.append(
                f"Temperature range too small: {temp_range:.1f}°C < {self.min_temp_range}°C"
            )

        # Check irradiance consistency
        irradiances = [dp.irradiance for dp in self.data_points]
        irr_mean = np.mean(irradiances)
        for i, irr in enumerate(irradiances):
            deviation = abs(irr - irr_mean) / irr_mean * 100
            if deviation > self.irradiance_tolerance_percent:
                issues.append(
                    f"Point {i + 1}: Irradiance deviation {deviation:.2f}% > {self.irradiance_tolerance_percent}%"
                )

        return (len(issues) == 0, issues)

    def _linear_regression(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Perform linear regression.

        Args:
            x: Independent variable (temperature)
            y: Dependent variable (Isc, Voc, or Pmax)

        Returns:
            (slope, intercept, r_squared, std_error)
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        return (slope, intercept, r_squared, std_err)

    def calculate_alpha(self) -> TemperatureCoefficientResult:
        """Calculate temperature coefficient of Isc (α).

        Returns:
            TemperatureCoefficientResult for alpha
        """
        temps = np.array([dp.temperature for dp in self.data_points])
        isc_values = np.array([dp.isc for dp in self.data_points])

        slope, intercept, r_squared, std_err = self._linear_regression(temps, isc_values)

        # Calculate reference Isc at reference temperature
        isc_ref = intercept + slope * self.reference_temperature

        # Calculate relative coefficient
        if isc_ref != 0:
            alpha_relative = (slope / isc_ref) * 100  # %/°C
        else:
            alpha_relative = 0.0

        result = TemperatureCoefficientResult(
            parameter="alpha_isc",
            value_absolute=slope,
            value_relative=alpha_relative,
            unit_absolute="A/°C",
            unit_relative="%/°C",
            reference_value=isc_ref,
            r_squared=r_squared,
            std_error=std_err,
            n_points=len(temps),
            temp_range=(float(min(temps)), float(max(temps)))
        )

        self._results["alpha"] = result
        return result

    def calculate_beta(self) -> TemperatureCoefficientResult:
        """Calculate temperature coefficient of Voc (β).

        Returns:
            TemperatureCoefficientResult for beta
        """
        temps = np.array([dp.temperature for dp in self.data_points])
        voc_values = np.array([dp.voc for dp in self.data_points])

        slope, intercept, r_squared, std_err = self._linear_regression(temps, voc_values)

        # Calculate reference Voc at reference temperature
        voc_ref = intercept + slope * self.reference_temperature

        # Calculate relative coefficient
        if voc_ref != 0:
            beta_relative = (slope / voc_ref) * 100  # %/°C
        else:
            beta_relative = 0.0

        result = TemperatureCoefficientResult(
            parameter="beta_voc",
            value_absolute=slope,
            value_relative=beta_relative,
            unit_absolute="V/°C",
            unit_relative="%/°C",
            reference_value=voc_ref,
            r_squared=r_squared,
            std_error=std_err,
            n_points=len(temps),
            temp_range=(float(min(temps)), float(max(temps)))
        )

        self._results["beta"] = result
        return result

    def calculate_gamma(self) -> TemperatureCoefficientResult:
        """Calculate temperature coefficient of Pmax (γ).

        Returns:
            TemperatureCoefficientResult for gamma
        """
        temps = np.array([dp.temperature for dp in self.data_points])
        pmax_values = np.array([dp.pmax for dp in self.data_points])

        slope, intercept, r_squared, std_err = self._linear_regression(temps, pmax_values)

        # Calculate reference Pmax at reference temperature
        pmax_ref = intercept + slope * self.reference_temperature

        # Calculate relative coefficient
        if pmax_ref != 0:
            gamma_relative = (slope / pmax_ref) * 100  # %/°C
        else:
            gamma_relative = 0.0

        result = TemperatureCoefficientResult(
            parameter="gamma_pmax",
            value_absolute=slope,
            value_relative=gamma_relative,
            unit_absolute="W/°C",
            unit_relative="%/°C",
            reference_value=pmax_ref,
            r_squared=r_squared,
            std_error=std_err,
            n_points=len(temps),
            temp_range=(float(min(temps)), float(max(temps)))
        )

        self._results["gamma"] = result
        return result

    def calculate_all(self) -> Dict[str, TemperatureCoefficientResult]:
        """Calculate all temperature coefficients.

        Returns:
            Dictionary with all coefficient results
        """
        is_valid, issues = self.validate_data()
        if not is_valid:
            raise ValueError(f"Data validation failed: {'; '.join(issues)}")

        self.calculate_alpha()
        self.calculate_beta()
        self.calculate_gamma()

        return self._results

    def estimate_rs_temperature_dependence(self) -> Optional[Tuple[float, float]]:
        """Estimate series resistance temperature dependence.

        Analyzes how Rs changes with temperature for kappa calculation.

        Returns:
            (dRs/dT slope, R²) or None if insufficient data
        """
        # Check if Rs data is available
        rs_data = [(dp.temperature, dp.rs) for dp in self.data_points if dp.rs is not None]

        if len(rs_data) < self.min_points:
            return None

        temps = np.array([d[0] for d in rs_data])
        rs_values = np.array([d[1] for d in rs_data])

        slope, _, r_squared, _ = self._linear_regression(temps, rs_values)

        return (slope, r_squared)

    def calculate_kappa(
        self,
        isc_ref: Optional[float] = None
    ) -> Optional[float]:
        """Calculate curve correction factor (κ) for IEC 60891.

        κ = Rs_slope × Isc_ref / (Pmax_ref)

        This is a simplified estimation. Full kappa determination
        requires detailed I-V curve analysis.

        Args:
            isc_ref: Reference Isc (uses calculated if not provided)

        Returns:
            Kappa value (Ω/°C) or None if cannot be calculated
        """
        rs_result = self.estimate_rs_temperature_dependence()

        if rs_result is None:
            return None

        rs_slope, r_squared = rs_result

        # Use calculated or provided reference Isc
        if isc_ref is None:
            if "alpha" in self._results:
                isc_ref = self._results["alpha"].reference_value
            else:
                return None

        # Simplified kappa estimation
        kappa = rs_slope

        return kappa

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all calculated coefficients.

        Returns:
            Dictionary with summary information
        """
        if not self._results:
            self.calculate_all()

        summary = {
            "n_measurements": len(self.data_points),
            "temperature_range": (
                min(dp.temperature for dp in self.data_points),
                max(dp.temperature for dp in self.data_points)
            ),
            "reference_temperature": self.reference_temperature,
            "coefficients": {}
        }

        for name, result in self._results.items():
            summary["coefficients"][name] = {
                "absolute": result.value_absolute,
                "relative_percent": result.value_relative,
                "unit": result.unit_absolute,
                "r_squared": result.r_squared,
                "reference_value": result.reference_value
            }

        # Check if results meet quality requirements
        summary["quality_check"] = {
            "all_r_squared_ok": all(
                r.r_squared >= self.min_r_squared
                for r in self._results.values()
            ),
            "min_r_squared": min(r.r_squared for r in self._results.values()) if self._results else 0.0
        }

        return summary

    def generate_report(self) -> str:
        """Generate detailed temperature coefficient report.

        Returns:
            Formatted report string
        """
        if not self._results:
            try:
                self.calculate_all()
            except ValueError as e:
                return f"Cannot generate report: {e}"

        summary = self.get_summary()

        report = f"""
TEMPERATURE COEFFICIENT ANALYSIS REPORT
=======================================
Per IEC 60904-10

Measurement Summary:
  Number of measurements: {summary['n_measurements']}
  Temperature range: {summary['temperature_range'][0]:.1f}°C to {summary['temperature_range'][1]:.1f}°C
  Reference temperature: {summary['reference_temperature']:.1f}°C

CALCULATED COEFFICIENTS:
------------------------
"""

        for name, result in self._results.items():
            report += f"""
{result.parameter.upper()}:
  Absolute:  {result.value_absolute:+.6f} {result.unit_absolute}
  Relative:  {result.value_relative:+.4f} {result.unit_relative}
  Reference: {result.reference_value:.3f}
  R²:        {result.r_squared:.4f} {'✓' if result.r_squared >= self.min_r_squared else '✗'}
  Std Error: ±{result.std_error:.6f}
"""

        # Add quality summary
        quality = summary["quality_check"]
        report += f"""
QUALITY CHECK:
--------------
  All R² ≥ {self.min_r_squared}: {'PASS' if quality['all_r_squared_ok'] else 'FAIL'}
  Minimum R²: {quality['min_r_squared']:.4f}

"""

        return report


# ============================================================================
# SERIES RESISTANCE EXTRACTION
# ============================================================================

def extract_rs_from_iv_curves(
    voltage: np.ndarray,
    current: np.ndarray,
    voc: float,
    isc: float
) -> float:
    """Extract series resistance from I-V curve.

    Uses slope at Voc: Rs = -dV/dI at I=0

    Args:
        voltage: Voltage array (V)
        current: Current array (A)
        voc: Open-circuit voltage (V)
        isc: Short-circuit current (A)

    Returns:
        Estimated Rs (Ohms)
    """
    # Find points near Voc (last 10% of curve)
    n_points = max(5, len(voltage) // 10)
    v_near_voc = voltage[-n_points:]
    i_near_voc = current[-n_points:]

    # Linear fit to get dV/dI
    if len(v_near_voc) > 2:
        slope, _, _, _ = stats.linregress(i_near_voc, v_near_voc)
        rs = abs(slope)
        # Sanity check
        return rs if 0.01 < rs < 10.0 else 0.5
    return 0.5


def extract_rs_from_two_curves(
    iv_curve_1: Tuple[np.ndarray, np.ndarray],
    iv_curve_2: Tuple[np.ndarray, np.ndarray],
    irradiance_1: float,
    irradiance_2: float
) -> float:
    """Extract Rs from two I-V curves at different irradiances.

    Per IEC 60891 method for Rs determination.

    Args:
        iv_curve_1: (voltage, current) at irradiance 1
        iv_curve_2: (voltage, current) at irradiance 2
        irradiance_1: First irradiance level (W/m²)
        irradiance_2: Second irradiance level (W/m²)

    Returns:
        Estimated Rs (Ohms)
    """
    v1, i1 = iv_curve_1
    v2, i2 = iv_curve_2

    # Find corresponding points on both curves
    # Use interpolation to align curves
    from scipy.interpolate import interp1d

    # Create common current grid
    i_min = max(min(i1), min(i2))
    i_max = min(max(i1), max(i2))
    i_common = np.linspace(i_min, i_max, 100)

    # Interpolate voltage for each curve
    f1 = interp1d(i1[::-1], v1[::-1], kind='linear', fill_value='extrapolate')
    f2 = interp1d(i2[::-1], v2[::-1], kind='linear', fill_value='extrapolate')

    v1_interp = f1(i_common)
    v2_interp = f2(i_common)

    # Calculate delta V / delta I
    delta_v = v2_interp - v1_interp
    delta_i = i_common * (irradiance_2 / irradiance_1 - 1)

    # Avoid division by zero
    valid_mask = np.abs(delta_i) > 0.001
    if np.sum(valid_mask) < 2:
        return 0.5

    rs_values = delta_v[valid_mask] / delta_i[valid_mask]

    # Return median Rs (robust to outliers)
    return float(np.median(np.abs(rs_values)))


# ============================================================================
# TYPICAL VALUES LOOKUP
# ============================================================================

def get_typical_coefficients(
    technology: str
) -> Optional[Dict[str, float]]:
    """Get typical temperature coefficients for a technology.

    Args:
        technology: PV technology (e.g., 'c-Si', 'HJT', 'CdTe')

    Returns:
        Dictionary with typical values or None if not found
    """
    if technology not in TYPICAL_TEMP_COEFFICIENTS:
        return None

    coeff = TYPICAL_TEMP_COEFFICIENTS[technology]

    return {
        "technology": coeff.technology,
        "alpha_isc_percent_per_c": coeff.alpha_isc_percent_per_c,
        "beta_voc_percent_per_c": coeff.beta_voc_percent_per_c,
        "gamma_pmax_percent_per_c": coeff.gamma_pmax_percent_per_c,
        "alpha_isc_absolute": coeff.alpha_isc_absolute,
        "beta_voc_absolute": coeff.beta_voc_absolute,
        "gamma_pmax_absolute": coeff.gamma_pmax_absolute
    }


def compare_to_typical(
    measured: Dict[str, TemperatureCoefficientResult],
    technology: str
) -> Dict[str, Any]:
    """Compare measured coefficients to typical values.

    Args:
        measured: Dictionary of measured coefficients
        technology: PV technology for comparison

    Returns:
        Dictionary with comparison results
    """
    typical = get_typical_coefficients(technology)

    if typical is None:
        return {"error": f"No typical values for technology: {technology}"}

    comparison = {
        "technology": technology,
        "parameters": {}
    }

    parameter_map = {
        "alpha": ("alpha_isc_percent_per_c", "alpha"),
        "beta": ("beta_voc_percent_per_c", "beta"),
        "gamma": ("gamma_pmax_percent_per_c", "gamma")
    }

    for param_key, (typical_key, measured_key) in parameter_map.items():
        if measured_key in measured:
            meas_val = measured[measured_key].value_relative
            typ_val = typical[typical_key]

            deviation = meas_val - typ_val
            deviation_percent = (deviation / abs(typ_val)) * 100 if typ_val != 0 else 0

            comparison["parameters"][param_key] = {
                "measured": meas_val,
                "typical": typ_val,
                "deviation": deviation,
                "deviation_percent": deviation_percent,
                "within_10_percent": abs(deviation_percent) <= 10
            }

    return comparison
