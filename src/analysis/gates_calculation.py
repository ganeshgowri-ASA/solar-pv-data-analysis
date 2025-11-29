"""Pass/Fail Gate Calculation Module.

Implements comprehensive pass/fail determination for PV module testing
with configurable tolerance gates, deviation analysis, and conformity
statements per IEC 17025 requirements.

Key features:
- Configurable tolerance gates for all parameters
- Automatic deviation flagging with severity levels
- Guard band calculations for conformity statements
- Statistical process control (SPC) support
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime


class DeviationSeverity(Enum):
    """Severity levels for deviation flags."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OUT_OF_SPEC = "out_of_spec"


class ConformityDecision(Enum):
    """Conformity decision states per ISO/IEC 17025."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL_PASS = "conditional_pass"
    CONDITIONAL_FAIL = "conditional_fail"
    INDETERMINATE = "indeterminate"


class GateResult(NamedTuple):
    """Result of a single gate check."""
    parameter: str
    value: float
    target: float
    tolerance_lower: float
    tolerance_upper: float
    deviation: float
    deviation_percent: float
    passed: bool
    severity: DeviationSeverity
    unit: str


@dataclass
class ToleranceGate:
    """Tolerance gate for a single parameter.

    Supports symmetric tolerances (±%) and asymmetric tolerances.
    """
    parameter_name: str
    target_value: float
    unit: str = ""

    # Tolerance specification
    tolerance_percent: Optional[float] = None  # Symmetric ±%
    tolerance_absolute: Optional[float] = None  # Symmetric ±abs
    tolerance_lower: Optional[float] = None  # Asymmetric lower limit
    tolerance_upper: Optional[float] = None  # Asymmetric upper limit

    # Severity thresholds (as % of tolerance)
    warning_threshold_percent: float = 80.0  # 80% of tolerance
    critical_threshold_percent: float = 95.0  # 95% of tolerance

    # Guard band for conformity statement (% of uncertainty)
    guard_band_factor: float = 1.0  # Multiplier for uncertainty

    # Optional reference to specification
    specification_ref: str = ""

    def __post_init__(self):
        """Calculate absolute tolerance limits."""
        if self.tolerance_percent is not None:
            half_tol = self.target_value * (self.tolerance_percent / 100.0)
            self._lower = self.target_value - half_tol
            self._upper = self.target_value + half_tol
        elif self.tolerance_absolute is not None:
            self._lower = self.target_value - self.tolerance_absolute
            self._upper = self.target_value + self.tolerance_absolute
        elif self.tolerance_lower is not None and self.tolerance_upper is not None:
            self._lower = self.tolerance_lower
            self._upper = self.tolerance_upper
        else:
            # No tolerance specified - use 0 (exact match required)
            self._lower = self.target_value
            self._upper = self.target_value

    @property
    def lower_limit(self) -> float:
        """Return lower tolerance limit."""
        return self._lower

    @property
    def upper_limit(self) -> float:
        """Return upper tolerance limit."""
        return self._upper

    @property
    def tolerance_range(self) -> float:
        """Return total tolerance range."""
        return self._upper - self._lower

    def check(self, value: float) -> GateResult:
        """Check if value passes gate.

        Args:
            value: Measured value to check

        Returns:
            GateResult with detailed pass/fail information
        """
        # Calculate deviation
        deviation = value - self.target_value
        if self.target_value != 0:
            deviation_percent = (deviation / self.target_value) * 100.0
        else:
            deviation_percent = float('inf') if deviation != 0 else 0.0

        # Check if within tolerance
        passed = self._lower <= value <= self._upper

        # Determine severity based on position within tolerance band
        half_range = self.tolerance_range / 2
        if half_range > 0:
            position_percent = (abs(deviation) / half_range) * 100.0
        else:
            position_percent = 100.0 if deviation != 0 else 0.0

        if not passed:
            severity = DeviationSeverity.OUT_OF_SPEC
        elif position_percent >= self.critical_threshold_percent:
            severity = DeviationSeverity.CRITICAL
        elif position_percent >= self.warning_threshold_percent:
            severity = DeviationSeverity.WARNING
        else:
            severity = DeviationSeverity.NORMAL

        return GateResult(
            parameter=self.parameter_name,
            value=value,
            target=self.target_value,
            tolerance_lower=self._lower,
            tolerance_upper=self._upper,
            deviation=deviation,
            deviation_percent=deviation_percent,
            passed=passed,
            severity=severity,
            unit=self.unit
        )

    def check_with_uncertainty(
        self,
        value: float,
        uncertainty: float
    ) -> Tuple[GateResult, ConformityDecision]:
        """Check value with uncertainty for conformity statement.

        Implements guard band approach per ILAC-G8 guidelines.

        Args:
            value: Measured value
            uncertainty: Expanded uncertainty (k=2)

        Returns:
            (GateResult, ConformityDecision)
        """
        basic_result = self.check(value)

        # Calculate guard band
        guard_band = uncertainty * self.guard_band_factor

        # Adjusted limits with guard band (inward)
        adj_lower = self._lower + guard_band
        adj_upper = self._upper - guard_band

        # Determine conformity decision
        if value >= adj_lower and value <= adj_upper:
            # Clear pass - value within guarded limits
            decision = ConformityDecision.PASS
        elif value < self._lower or value > self._upper:
            # Clear fail - value outside tolerance
            if (value + uncertainty >= self._lower and
                    value - uncertainty <= self._upper):
                # But uncertainty overlaps tolerance
                decision = ConformityDecision.CONDITIONAL_FAIL
            else:
                decision = ConformityDecision.FAIL
        else:
            # Value in guard band region
            if value < adj_lower:
                # Near lower limit
                decision = ConformityDecision.CONDITIONAL_PASS
            else:
                # Near upper limit
                decision = ConformityDecision.CONDITIONAL_PASS

        return (basic_result, decision)


@dataclass
class GateSet:
    """Collection of tolerance gates for comprehensive testing."""
    name: str
    description: str = ""
    gates: Dict[str, ToleranceGate] = field(default_factory=dict)

    def add_gate(self, gate: ToleranceGate):
        """Add a tolerance gate."""
        self.gates[gate.parameter_name] = gate

    def check_all(self, values: Dict[str, float]) -> Dict[str, GateResult]:
        """Check all values against their gates.

        Args:
            values: Dictionary mapping parameter names to values

        Returns:
            Dictionary mapping parameter names to GateResults
        """
        results = {}
        for param_name, value in values.items():
            if param_name in self.gates:
                results[param_name] = self.gates[param_name].check(value)
        return results

    def overall_pass(self, results: Dict[str, GateResult]) -> bool:
        """Check if all gates passed."""
        return all(r.passed for r in results.values())

    def get_failures(self, results: Dict[str, GateResult]) -> List[GateResult]:
        """Get list of failed gates."""
        return [r for r in results.values() if not r.passed]

    def get_warnings(self, results: Dict[str, GateResult]) -> List[GateResult]:
        """Get list of warning-level results."""
        return [
            r for r in results.values()
            if r.severity in (DeviationSeverity.WARNING, DeviationSeverity.CRITICAL)
        ]


# ============================================================================
# STANDARD GATE SETS FOR PV TESTING
# ============================================================================

def create_power_tolerance_gates(
    rated_power: float,
    tolerance_percent: float = 3.0
) -> GateSet:
    """Create standard power tolerance gates.

    Args:
        rated_power: Rated module power (Wp)
        tolerance_percent: Tolerance percentage (default ±3%)

    Returns:
        GateSet for power parameters
    """
    gate_set = GateSet(
        name="Power Tolerance",
        description=f"Power rating tolerances at ±{tolerance_percent}%"
    )

    gate_set.add_gate(ToleranceGate(
        parameter_name="Pmax",
        target_value=rated_power,
        unit="W",
        tolerance_percent=tolerance_percent,
        specification_ref="IEC 61215"
    ))

    return gate_set


def create_iv_parameter_gates(
    isc_rated: float,
    voc_rated: float,
    pmax_rated: float,
    ff_rated: float = 0.80,
    isc_tolerance: float = 5.0,
    voc_tolerance: float = 3.0,
    pmax_tolerance: float = 3.0,
    ff_tolerance: float = 5.0
) -> GateSet:
    """Create comprehensive I-V parameter gates.

    Args:
        isc_rated: Rated Isc (A)
        voc_rated: Rated Voc (V)
        pmax_rated: Rated Pmax (W)
        ff_rated: Rated fill factor
        *_tolerance: Tolerance percentages

    Returns:
        GateSet for I-V parameters
    """
    gate_set = GateSet(
        name="I-V Parameters",
        description="I-V curve parameter tolerances"
    )

    gate_set.add_gate(ToleranceGate(
        parameter_name="Isc",
        target_value=isc_rated,
        unit="A",
        tolerance_percent=isc_tolerance
    ))

    gate_set.add_gate(ToleranceGate(
        parameter_name="Voc",
        target_value=voc_rated,
        unit="V",
        tolerance_percent=voc_tolerance
    ))

    gate_set.add_gate(ToleranceGate(
        parameter_name="Pmax",
        target_value=pmax_rated,
        unit="W",
        tolerance_percent=pmax_tolerance
    ))

    gate_set.add_gate(ToleranceGate(
        parameter_name="FF",
        target_value=ff_rated,
        unit="",
        tolerance_percent=ff_tolerance
    ))

    return gate_set


def create_degradation_gates(
    initial_pmax: float,
    max_degradation_percent: float = 5.0
) -> GateSet:
    """Create degradation tolerance gates.

    Args:
        initial_pmax: Initial power measurement (W)
        max_degradation_percent: Maximum allowed degradation (%)

    Returns:
        GateSet for degradation assessment
    """
    min_power = initial_pmax * (1 - max_degradation_percent / 100.0)

    gate_set = GateSet(
        name="Degradation Assessment",
        description=f"Maximum {max_degradation_percent}% degradation allowed"
    )

    gate_set.add_gate(ToleranceGate(
        parameter_name="Pmax",
        target_value=initial_pmax,
        unit="W",
        tolerance_lower=min_power,
        tolerance_upper=initial_pmax * 1.05,  # Allow small increase
        specification_ref="IEC 61215"
    ))

    return gate_set


# ============================================================================
# DEVIATION ANALYSIS
# ============================================================================

@dataclass
class DeviationFlag:
    """Detailed deviation flag for quality control."""
    parameter_name: str
    measured_value: float
    prescribed_value: float
    tolerance: float
    deviation_absolute: float
    deviation_percentage: float
    severity: DeviationSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""

    @property
    def is_critical(self) -> bool:
        """Check if deviation is critical."""
        return self.severity in (DeviationSeverity.CRITICAL, DeviationSeverity.OUT_OF_SPEC)


class DeviationAnalyzer:
    """Analyze deviations and generate flags."""

    def __init__(self, gate_set: GateSet):
        """Initialize with gate set.

        Args:
            gate_set: GateSet defining tolerances
        """
        self.gate_set = gate_set
        self.flags: List[DeviationFlag] = []

    def analyze(
        self,
        values: Dict[str, float],
        clear_previous: bool = True
    ) -> List[DeviationFlag]:
        """Analyze values and generate deviation flags.

        Args:
            values: Dictionary of parameter values
            clear_previous: Whether to clear previous flags

        Returns:
            List of deviation flags
        """
        if clear_previous:
            self.flags.clear()

        results = self.gate_set.check_all(values)

        for param_name, result in results.items():
            gate = self.gate_set.gates[param_name]

            flag = DeviationFlag(
                parameter_name=param_name,
                measured_value=result.value,
                prescribed_value=result.target,
                tolerance=gate.tolerance_range / 2,
                deviation_absolute=result.deviation,
                deviation_percentage=result.deviation_percent,
                severity=result.severity
            )

            self.flags.append(flag)

        return self.flags

    def get_critical_flags(self) -> List[DeviationFlag]:
        """Get critical deviation flags."""
        return [f for f in self.flags if f.is_critical]

    def get_summary(self) -> Dict[str, Any]:
        """Get deviation analysis summary."""
        total = len(self.flags)
        critical = len(self.get_critical_flags())
        warnings = len([f for f in self.flags if f.severity == DeviationSeverity.WARNING])
        normal = len([f for f in self.flags if f.severity == DeviationSeverity.NORMAL])
        out_of_spec = len([f for f in self.flags if f.severity == DeviationSeverity.OUT_OF_SPEC])

        return {
            "total_parameters": total,
            "out_of_spec": out_of_spec,
            "critical": critical,
            "warnings": warnings,
            "normal": normal,
            "overall_pass": out_of_spec == 0,
            "flags": self.flags
        }


# ============================================================================
# CONFORMITY STATEMENT GENERATOR
# ============================================================================

def generate_conformity_statement(
    results: Dict[str, GateResult],
    uncertainties: Optional[Dict[str, float]] = None,
    decision_rule: str = "simple"
) -> Dict[str, Any]:
    """Generate conformity statement per ISO/IEC 17025.

    Args:
        results: Dictionary of gate results
        uncertainties: Optional dictionary of expanded uncertainties
        decision_rule: "simple" or "guard_band"

    Returns:
        Dictionary with conformity statement details
    """
    passed_count = sum(1 for r in results.values() if r.passed)
    total_count = len(results)

    failures = [r for r in results.values() if not r.passed]
    warnings = [
        r for r in results.values()
        if r.severity in (DeviationSeverity.WARNING, DeviationSeverity.CRITICAL)
    ]

    if total_count == 0:
        overall = ConformityDecision.INDETERMINATE
    elif len(failures) == 0:
        overall = ConformityDecision.PASS
    else:
        overall = ConformityDecision.FAIL

    statement = {
        "conformity_decision": overall.value,
        "passed_parameters": passed_count,
        "total_parameters": total_count,
        "pass_rate_percent": (passed_count / total_count * 100) if total_count > 0 else 0,
        "decision_rule": decision_rule,
        "failures": [
            {
                "parameter": f.parameter,
                "value": f.value,
                "tolerance_lower": f.tolerance_lower,
                "tolerance_upper": f.tolerance_upper,
                "deviation_percent": f.deviation_percent
            }
            for f in failures
        ],
        "warnings": [
            {
                "parameter": w.parameter,
                "value": w.value,
                "severity": w.severity.value
            }
            for w in warnings
        ],
        "timestamp": datetime.now().isoformat()
    }

    # Generate human-readable statement
    if overall == ConformityDecision.PASS:
        statement["statement_text"] = (
            f"All {total_count} tested parameters conform to specification."
        )
    elif overall == ConformityDecision.FAIL:
        param_list = ", ".join(f.parameter for f in failures)
        statement["statement_text"] = (
            f"{len(failures)} of {total_count} parameters do not conform to "
            f"specification: {param_list}."
        )
    else:
        statement["statement_text"] = "Unable to determine conformity."

    return statement


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_cpk(
    values: np.ndarray,
    lower_limit: float,
    upper_limit: float
) -> float:
    """Calculate process capability index Cpk.

    Args:
        values: Array of measured values
        lower_limit: Lower specification limit
        upper_limit: Upper specification limit

    Returns:
        Cpk value (>1.33 generally desired)
    """
    if len(values) < 2:
        return 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if std == 0:
        return float('inf') if lower_limit <= mean <= upper_limit else 0.0

    cpu = (upper_limit - mean) / (3 * std)
    cpl = (mean - lower_limit) / (3 * std)

    return min(cpu, cpl)


def calculate_yield(
    values: np.ndarray,
    lower_limit: float,
    upper_limit: float
) -> float:
    """Calculate yield (% within specification).

    Args:
        values: Array of measured values
        lower_limit: Lower specification limit
        upper_limit: Upper specification limit

    Returns:
        Yield percentage (0-100)
    """
    if len(values) == 0:
        return 0.0

    within_spec = np.sum((values >= lower_limit) & (values <= upper_limit))
    return (within_spec / len(values)) * 100.0


def power_bin_classification(
    power: float,
    rated_power: float,
    bin_size_w: float = 5.0,
    num_bins_positive: int = 3,
    num_bins_negative: int = 3
) -> Tuple[str, float, float]:
    """Classify module into power bin.

    Args:
        power: Measured power (W)
        rated_power: Rated power (W)
        bin_size_w: Size of each bin (W)
        num_bins_positive: Number of positive bins
        num_bins_negative: Number of negative bins

    Returns:
        (bin_label, bin_lower, bin_upper)
    """
    deviation = power - rated_power

    for i in range(-num_bins_negative, num_bins_positive + 1):
        lower = rated_power + (i * bin_size_w)
        upper = rated_power + ((i + 1) * bin_size_w)

        if lower <= power < upper:
            if i == 0:
                label = "0"
            elif i > 0:
                label = f"+{i}"
            else:
                label = str(i)

            return (label, lower, upper)

    # Outside defined bins
    if deviation >= 0:
        return (f"+{num_bins_positive}+", rated_power + num_bins_positive * bin_size_w, float('inf'))
    else:
        return (f"-{num_bins_negative}-", float('-inf'), rated_power - num_bins_negative * bin_size_w)
