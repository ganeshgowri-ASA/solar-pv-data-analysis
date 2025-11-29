"""GUM Uncertainty Propagation Module.

Implements uncertainty analysis per GUM (Guide to the Expression of
Uncertainty in Measurement) for PV testing applications.

Key features:
- Type A (statistical) and Type B (other) uncertainty evaluation
- Sensitivity coefficient calculation
- Combined standard uncertainty
- Expanded uncertainty with coverage factors
- Uncertainty budget generation
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.iec_standards import UNCERTAINTY_COMPONENTS


class UncertaintyType(Enum):
    """Type of uncertainty evaluation."""
    TYPE_A = "type_a"  # Statistical evaluation
    TYPE_B = "type_b"  # Other methods


class Distribution(Enum):
    """Probability distribution types."""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"


class CorrelationType(Enum):
    """Correlation between uncertainty sources."""
    INDEPENDENT = "independent"  # Uncorrelated
    POSITIVE = "positive"  # Positively correlated
    NEGATIVE = "negative"  # Negatively correlated


class UncertaintyResult(NamedTuple):
    """Result of uncertainty calculation."""
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float
    confidence_level_percent: float
    degrees_of_freedom: int


@dataclass
class UncertaintyComponent:
    """Single uncertainty component for budget calculation."""
    name: str
    symbol: str
    description: str = ""

    # Value specification
    value: float = 0.0  # Absolute or relative value
    is_relative: bool = False  # True if value is in %
    unit: str = ""

    # Distribution and type
    uncertainty_type: UncertaintyType = UncertaintyType.TYPE_B
    distribution: Distribution = Distribution.NORMAL

    # Coverage and divisor
    coverage_factor_input: float = 1.0  # k-factor for input value
    divisor: float = 1.0  # Divisor for standard uncertainty

    # Sensitivity coefficient
    sensitivity_coefficient: float = 1.0

    # Correlation
    correlation_type: CorrelationType = CorrelationType.INDEPENDENT
    correlated_with: List[str] = field(default_factory=list)

    # Degrees of freedom (for Type A)
    degrees_of_freedom: int = 50  # Default to large for Type B

    @property
    def standard_uncertainty(self) -> float:
        """Calculate standard uncertainty.

        For Type A: u = s / sqrt(n) (provided directly)
        For Type B: u = value / (coverage_factor * divisor)
        """
        if self.distribution == Distribution.NORMAL:
            return self.value / (self.coverage_factor_input * self.divisor)
        elif self.distribution == Distribution.RECTANGULAR:
            # Rectangular: divisor is sqrt(3)
            return self.value / np.sqrt(3)
        elif self.distribution == Distribution.TRIANGULAR:
            # Triangular: divisor is sqrt(6)
            return self.value / np.sqrt(6)
        elif self.distribution == Distribution.U_SHAPED:
            # U-shaped: divisor is sqrt(2)
            return self.value / np.sqrt(2)
        else:
            return self.value / self.divisor

    @property
    def contribution(self) -> float:
        """Calculate contribution to combined uncertainty.

        contribution = (sensitivity_coefficient × standard_uncertainty)²
        """
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget for a measurement."""

    # Identification
    measurement_name: str
    measurement_value: float
    measurement_unit: str

    # Components
    components: List[UncertaintyComponent] = field(default_factory=list)

    # Calculation results
    _combined_uncertainty: Optional[float] = None
    _expanded_uncertainty: Optional[float] = None
    _effective_dof: Optional[float] = None

    # Configuration
    coverage_factor: float = 2.0  # Default k=2 for ~95% confidence
    custom_confidence_level: Optional[float] = None

    def add_component(self, component: UncertaintyComponent):
        """Add an uncertainty component."""
        self.components.append(component)

    def add_type_a_component(
        self,
        name: str,
        symbol: str,
        std_deviation: float,
        n_measurements: int,
        description: str = ""
    ):
        """Add Type A (statistical) uncertainty component.

        Args:
            name: Component name
            symbol: Symbol for equations
            std_deviation: Standard deviation of measurements
            n_measurements: Number of measurements
            description: Optional description
        """
        # Standard uncertainty of mean = s / sqrt(n)
        std_uncertainty = std_deviation / np.sqrt(n_measurements)

        self.components.append(UncertaintyComponent(
            name=name,
            symbol=symbol,
            description=description,
            value=std_uncertainty,
            uncertainty_type=UncertaintyType.TYPE_A,
            distribution=Distribution.NORMAL,
            coverage_factor_input=1.0,
            divisor=1.0,
            degrees_of_freedom=n_measurements - 1
        ))

    def add_type_b_component(
        self,
        name: str,
        symbol: str,
        value: float,
        distribution: Distribution = Distribution.RECTANGULAR,
        coverage_factor: float = 1.0,
        sensitivity: float = 1.0,
        is_relative: bool = False,
        description: str = ""
    ):
        """Add Type B uncertainty component.

        Args:
            name: Component name
            symbol: Symbol for equations
            value: Uncertainty value (half-width for rectangular)
            distribution: Probability distribution
            coverage_factor: Coverage factor of input value
            sensitivity: Sensitivity coefficient
            is_relative: True if value is in percentage
            description: Optional description
        """
        self.components.append(UncertaintyComponent(
            name=name,
            symbol=symbol,
            description=description,
            value=value,
            is_relative=is_relative,
            uncertainty_type=UncertaintyType.TYPE_B,
            distribution=distribution,
            coverage_factor_input=coverage_factor,
            sensitivity_coefficient=sensitivity
        ))

    def calculate_combined_uncertainty(self) -> float:
        """Calculate combined standard uncertainty.

        u_c = sqrt(sum of all contributions)

        Returns:
            Combined standard uncertainty
        """
        if not self.components:
            return 0.0

        # Sum of squared contributions (assuming independence)
        variance_sum = sum(c.contribution for c in self.components)

        self._combined_uncertainty = np.sqrt(variance_sum)
        return self._combined_uncertainty

    def calculate_effective_dof(self) -> float:
        """Calculate effective degrees of freedom using Welch-Satterthwaite.

        v_eff = u_c^4 / sum(u_i^4 / v_i)

        Returns:
            Effective degrees of freedom
        """
        if self._combined_uncertainty is None:
            self.calculate_combined_uncertainty()

        u_c = self._combined_uncertainty
        if u_c == 0:
            return float('inf')

        numerator = u_c ** 4

        denominator = sum(
            (c.sensitivity_coefficient * c.standard_uncertainty) ** 4 / c.degrees_of_freedom
            for c in self.components
            if c.degrees_of_freedom > 0
        )

        if denominator == 0:
            return float('inf')

        self._effective_dof = numerator / denominator
        return self._effective_dof

    def calculate_expanded_uncertainty(
        self,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate expanded uncertainty.

        U = k × u_c

        Args:
            confidence_level: Desired confidence level (0-1)

        Returns:
            (expanded_uncertainty, coverage_factor)
        """
        if self._combined_uncertainty is None:
            self.calculate_combined_uncertainty()

        if self._effective_dof is None:
            self.calculate_effective_dof()

        # Get coverage factor from t-distribution
        if self._effective_dof >= 1000:
            # Use normal distribution for large DOF
            k = stats.norm.ppf((1 + confidence_level) / 2)
        else:
            # Use t-distribution
            k = stats.t.ppf((1 + confidence_level) / 2, df=self._effective_dof)

        self.coverage_factor = k
        self._expanded_uncertainty = k * self._combined_uncertainty

        return (self._expanded_uncertainty, k)

    def get_result(self) -> UncertaintyResult:
        """Get complete uncertainty result.

        Returns:
            UncertaintyResult with all calculated values
        """
        u_c = self.calculate_combined_uncertainty()
        v_eff = self.calculate_effective_dof()
        u_exp, k = self.calculate_expanded_uncertainty()

        return UncertaintyResult(
            standard_uncertainty=u_c,
            expanded_uncertainty=u_exp,
            coverage_factor=k,
            confidence_level_percent=95.0,
            degrees_of_freedom=int(min(v_eff, 1000))
        )

    def get_relative_uncertainty(self) -> Tuple[float, float]:
        """Get uncertainty as percentage of measurement value.

        Returns:
            (relative_combined_%, relative_expanded_%)
        """
        if self._combined_uncertainty is None:
            self.calculate_combined_uncertainty()
        if self._expanded_uncertainty is None:
            self.calculate_expanded_uncertainty()

        if self.measurement_value == 0:
            return (0.0, 0.0)

        rel_combined = (self._combined_uncertainty / self.measurement_value) * 100
        rel_expanded = (self._expanded_uncertainty / self.measurement_value) * 100

        return (rel_combined, rel_expanded)

    def generate_budget_table(self) -> List[Dict[str, Any]]:
        """Generate uncertainty budget table.

        Returns:
            List of dictionaries with budget entries
        """
        if self._combined_uncertainty is None:
            self.calculate_combined_uncertainty()

        table = []
        for c in self.components:
            entry = {
                "name": c.name,
                "symbol": c.symbol,
                "type": c.uncertainty_type.value,
                "distribution": c.distribution.value,
                "value": c.value,
                "standard_uncertainty": c.standard_uncertainty,
                "sensitivity": c.sensitivity_coefficient,
                "contribution": np.sqrt(c.contribution),
                "contribution_squared": c.contribution,
                "dof": c.degrees_of_freedom
            }
            table.append(entry)

        return table

    def generate_report(self) -> str:
        """Generate detailed uncertainty report.

        Returns:
            Formatted report string
        """
        result = self.get_result()
        rel_unc = self.get_relative_uncertainty()
        budget = self.generate_budget_table()

        report = f"""
UNCERTAINTY BUDGET REPORT
=========================
Per GUM (JCGM 100:2008)

Measurement: {self.measurement_name}
Value: {self.measurement_value:.4f} {self.measurement_unit}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

UNCERTAINTY COMPONENTS:
-----------------------
"""
        # Table header
        report += f"{'Source':<30} {'Type':<8} {'Dist':<12} {'u(x)':<12} {'c_i':<8} {'c_i×u(x)':<12}\n"
        report += "-" * 90 + "\n"

        for entry in budget:
            report += (
                f"{entry['name']:<30} "
                f"{entry['type']:<8} "
                f"{entry['distribution']:<12} "
                f"{entry['standard_uncertainty']:<12.6f} "
                f"{entry['sensitivity']:<8.3f} "
                f"{entry['contribution']:<12.6f}\n"
            )

        report += "-" * 90 + "\n"

        report += f"""
COMBINED UNCERTAINTY:
---------------------
  Combined standard uncertainty (u_c):  {result.standard_uncertainty:.6f} {self.measurement_unit}
  Relative standard uncertainty:        {rel_unc[0]:.3f}%

  Effective degrees of freedom (v_eff): {result.degrees_of_freedom}

EXPANDED UNCERTAINTY:
---------------------
  Coverage factor (k):                  {result.coverage_factor:.2f}
  Confidence level:                     {result.confidence_level_percent:.0f}%
  Expanded uncertainty (U):             {result.expanded_uncertainty:.6f} {self.measurement_unit}
  Relative expanded uncertainty:        {rel_unc[1]:.3f}%

RESULT:
-------
  {self.measurement_name} = {self.measurement_value:.4f} ± {result.expanded_uncertainty:.4f} {self.measurement_unit}
  (k = {result.coverage_factor:.2f}, confidence level {result.confidence_level_percent:.0f}%)
"""

        return report


# ============================================================================
# STANDARD PV UNCERTAINTY BUDGETS
# ============================================================================

def create_pmax_uncertainty_budget(
    pmax_value: float,
    repeatability_std: float = 0.0,
    n_measurements: int = 1
) -> UncertaintyBudget:
    """Create standard Pmax uncertainty budget.

    Args:
        pmax_value: Measured Pmax (W)
        repeatability_std: Standard deviation of repeated measurements
        n_measurements: Number of measurements for repeatability

    Returns:
        UncertaintyBudget for Pmax
    """
    budget = UncertaintyBudget(
        measurement_name="Maximum Power (Pmax)",
        measurement_value=pmax_value,
        measurement_unit="W"
    )

    # Add standard components from IEC standards
    for comp in UNCERTAINTY_COMPONENTS:
        # Convert relative uncertainty to absolute
        abs_value = pmax_value * (comp.typical_value_percent / 100.0)

        if comp.distribution == "normal":
            dist = Distribution.NORMAL
            divisor = comp.divisor
        else:
            dist = Distribution.RECTANGULAR
            divisor = np.sqrt(3)

        budget.add_type_b_component(
            name=comp.name,
            symbol=comp.symbol,
            value=abs_value / divisor,  # Already standard uncertainty
            distribution=dist,
            description=comp.description
        )

    # Add repeatability if provided
    if repeatability_std > 0 and n_measurements > 1:
        budget.add_type_a_component(
            name="Repeatability",
            symbol="u_rep",
            std_deviation=repeatability_std,
            n_measurements=n_measurements,
            description="Type A uncertainty from repeated measurements"
        )

    return budget


def create_isc_uncertainty_budget(
    isc_value: float,
    calibration_uncertainty_percent: float = 1.0,
    spectral_mismatch_percent: float = 1.0,
    temperature_correction_percent: float = 0.3
) -> UncertaintyBudget:
    """Create Isc uncertainty budget.

    Args:
        isc_value: Measured Isc (A)
        calibration_uncertainty_percent: Reference cell calibration uncertainty
        spectral_mismatch_percent: Spectral mismatch correction uncertainty
        temperature_correction_percent: Temperature correction uncertainty

    Returns:
        UncertaintyBudget for Isc
    """
    budget = UncertaintyBudget(
        measurement_name="Short-Circuit Current (Isc)",
        measurement_value=isc_value,
        measurement_unit="A"
    )

    budget.add_type_b_component(
        name="Reference cell calibration",
        symbol="u_cal",
        value=isc_value * calibration_uncertainty_percent / 100 / 2,  # k=2 to k=1
        distribution=Distribution.NORMAL,
        description="Calibration uncertainty of reference cell (k=2)"
    )

    budget.add_type_b_component(
        name="Spectral mismatch",
        symbol="u_mm",
        value=isc_value * spectral_mismatch_percent / 100,
        distribution=Distribution.RECTANGULAR,
        description="Spectral mismatch correction uncertainty"
    )

    budget.add_type_b_component(
        name="Temperature correction",
        symbol="u_temp",
        value=isc_value * temperature_correction_percent / 100,
        distribution=Distribution.NORMAL,
        description="Temperature coefficient uncertainty"
    )

    budget.add_type_b_component(
        name="DAQ resolution",
        symbol="u_daq",
        value=0.001,  # 1 mA resolution
        distribution=Distribution.RECTANGULAR,
        description="Data acquisition resolution"
    )

    return budget


def create_voc_uncertainty_budget(
    voc_value: float,
    temperature_uncertainty_c: float = 1.0,
    beta_voc: float = -0.003  # V/°C per V
) -> UncertaintyBudget:
    """Create Voc uncertainty budget.

    Args:
        voc_value: Measured Voc (V)
        temperature_uncertainty_c: Temperature measurement uncertainty (°C)
        beta_voc: Temperature coefficient (%/°C)

    Returns:
        UncertaintyBudget for Voc
    """
    budget = UncertaintyBudget(
        measurement_name="Open-Circuit Voltage (Voc)",
        measurement_value=voc_value,
        measurement_unit="V"
    )

    # Temperature-related uncertainty
    temp_contribution = abs(beta_voc) * voc_value * temperature_uncertainty_c
    budget.add_type_b_component(
        name="Temperature measurement",
        symbol="u_temp",
        value=temp_contribution,
        distribution=Distribution.RECTANGULAR,
        description="Module temperature measurement"
    )

    # DAQ resolution
    budget.add_type_b_component(
        name="DAQ resolution",
        symbol="u_daq",
        value=0.01,  # 10 mV resolution
        distribution=Distribution.RECTANGULAR,
        description="Data acquisition resolution"
    )

    # Irradiance stability effect on Voc (weak dependence)
    budget.add_type_b_component(
        name="Irradiance stability",
        symbol="u_irr",
        value=voc_value * 0.001,  # 0.1% effect
        distribution=Distribution.RECTANGULAR,
        description="Effect of irradiance variations on Voc"
    )

    return budget


# ============================================================================
# UNCERTAINTY PROPAGATION UTILITIES
# ============================================================================

def propagate_uncertainty_product(
    value_a: float,
    uncertainty_a: float,
    value_b: float,
    uncertainty_b: float,
    correlation: float = 0.0
) -> Tuple[float, float]:
    """Propagate uncertainty for product C = A × B.

    Args:
        value_a: Value of A
        uncertainty_a: Standard uncertainty of A
        value_b: Value of B
        uncertainty_b: Standard uncertainty of B
        correlation: Correlation coefficient (-1 to 1)

    Returns:
        (product_value, product_uncertainty)
    """
    product = value_a * value_b

    # Relative uncertainties
    rel_a = uncertainty_a / value_a if value_a != 0 else 0
    rel_b = uncertainty_b / value_b if value_b != 0 else 0

    # Combined relative uncertainty
    rel_c = np.sqrt(rel_a ** 2 + rel_b ** 2 + 2 * correlation * rel_a * rel_b)

    return (product, abs(product) * rel_c)


def propagate_uncertainty_ratio(
    value_a: float,
    uncertainty_a: float,
    value_b: float,
    uncertainty_b: float,
    correlation: float = 0.0
) -> Tuple[float, float]:
    """Propagate uncertainty for ratio C = A / B.

    Args:
        value_a: Value of A (numerator)
        uncertainty_a: Standard uncertainty of A
        value_b: Value of B (denominator)
        uncertainty_b: Standard uncertainty of B
        correlation: Correlation coefficient (-1 to 1)

    Returns:
        (ratio_value, ratio_uncertainty)
    """
    if value_b == 0:
        return (float('inf'), float('inf'))

    ratio = value_a / value_b

    # Relative uncertainties
    rel_a = uncertainty_a / value_a if value_a != 0 else 0
    rel_b = uncertainty_b / value_b

    # Combined relative uncertainty (negative correlation for ratio)
    rel_c = np.sqrt(rel_a ** 2 + rel_b ** 2 - 2 * correlation * rel_a * rel_b)

    return (ratio, abs(ratio) * rel_c)


def propagate_uncertainty_sum(
    values: List[float],
    uncertainties: List[float],
    correlations: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Propagate uncertainty for sum C = A + B + ...

    Args:
        values: List of values
        uncertainties: List of standard uncertainties
        correlations: Optional correlation matrix

    Returns:
        (sum_value, sum_uncertainty)
    """
    total = sum(values)

    if correlations is None:
        # Assume independent
        variance = sum(u ** 2 for u in uncertainties)
    else:
        # Include correlations
        n = len(uncertainties)
        variance = 0.0
        for i in range(n):
            for j in range(n):
                variance += uncertainties[i] * uncertainties[j] * correlations[i, j]

    return (total, np.sqrt(variance))


def calculate_fill_factor_uncertainty(
    pmax: float,
    u_pmax: float,
    isc: float,
    u_isc: float,
    voc: float,
    u_voc: float
) -> Tuple[float, float]:
    """Calculate fill factor and its uncertainty.

    FF = Pmax / (Isc × Voc)

    Args:
        pmax: Maximum power (W)
        u_pmax: Uncertainty in Pmax
        isc: Short-circuit current (A)
        u_isc: Uncertainty in Isc
        voc: Open-circuit voltage (V)
        u_voc: Uncertainty in Voc

    Returns:
        (fill_factor, uncertainty)
    """
    # Calculate FF
    denominator = isc * voc
    if denominator == 0:
        return (0.0, 0.0)

    ff = pmax / denominator

    # Propagate uncertainty
    # FF = Pmax / (Isc × Voc)
    # Relative uncertainties
    rel_pmax = u_pmax / pmax if pmax != 0 else 0
    rel_isc = u_isc / isc if isc != 0 else 0
    rel_voc = u_voc / voc if voc != 0 else 0

    # Combined relative uncertainty (assuming independence)
    rel_ff = np.sqrt(rel_pmax ** 2 + rel_isc ** 2 + rel_voc ** 2)

    return (ff, ff * rel_ff)
