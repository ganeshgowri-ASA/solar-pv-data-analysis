"""Measurement Stabilization Module (IEC 60904-1).

Implements irradiance and temperature stabilization gates for ensuring
measurement conditions meet IEC requirements before data acquisition.

Key requirements:
- Irradiance stability: ±1% of target
- Temperature stability: ±1°C of target
- Minimum stabilization time: 60 seconds
- Minimum consecutive stable readings: 5
"""

from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.iec_standards import STABILIZATION_GATES, STC


class StabilizationStatus(Enum):
    """Status of stabilization check."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    WAITING = "waiting"
    TIMEOUT = "timeout"


@dataclass
class StabilizationReading:
    """Single reading during stabilization monitoring."""
    timestamp: datetime
    irradiance: float  # W/m²
    temperature: float  # °C

    @property
    def age_seconds(self) -> float:
        """Return age of reading in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class StabilizationResult(NamedTuple):
    """Result of stabilization check."""
    is_stable: bool
    irradiance_stable: bool
    temperature_stable: bool
    irradiance_deviation_percent: float
    temperature_deviation_c: float
    status: StabilizationStatus
    message: str
    stable_duration_s: float
    readings_count: int


@dataclass
class StabilizationGate:
    """Gate criteria for a single parameter."""
    parameter_name: str
    target_value: float
    tolerance: float  # Absolute tolerance for temperature, relative for irradiance
    is_relative: bool = False  # True for percentage tolerance
    unit: str = ""

    def is_within_gate(self, value: float) -> bool:
        """Check if value is within gate tolerance."""
        if self.is_relative:
            # Relative tolerance (percentage)
            tolerance_abs = self.target_value * (self.tolerance / 100.0)
        else:
            # Absolute tolerance
            tolerance_abs = self.tolerance

        return abs(value - self.target_value) <= tolerance_abs

    def deviation(self, value: float) -> float:
        """Calculate deviation from target."""
        if self.is_relative:
            return abs((value - self.target_value) / self.target_value) * 100.0
        return abs(value - self.target_value)


@dataclass
class StabilizationMonitor:
    """Monitor for measurement stabilization per IEC 60904-1.

    Tracks irradiance and temperature readings over time to determine
    when measurement conditions are stable enough for data acquisition.
    """
    # Target conditions
    target_irradiance: float = STC.irradiance
    target_temperature: float = STC.temperature

    # Tolerances
    irradiance_tolerance_percent: float = STABILIZATION_GATES.irradiance_tolerance_percent
    temperature_tolerance_c: float = STABILIZATION_GATES.temperature_tolerance_c

    # Timing requirements
    min_stabilization_time_s: float = STABILIZATION_GATES.min_stabilization_time_s
    min_stable_readings: int = STABILIZATION_GATES.min_stable_readings
    max_wait_time_s: float = 600.0  # Maximum wait time before timeout

    # Internal state
    readings: List[StabilizationReading] = field(default_factory=list)
    start_time: Optional[datetime] = None
    stable_since: Optional[datetime] = None

    # Gates
    irradiance_gate: StabilizationGate = field(init=False)
    temperature_gate: StabilizationGate = field(init=False)

    def __post_init__(self):
        """Initialize gates."""
        self.irradiance_gate = StabilizationGate(
            parameter_name="Irradiance",
            target_value=self.target_irradiance,
            tolerance=self.irradiance_tolerance_percent,
            is_relative=True,
            unit="W/m²"
        )
        self.temperature_gate = StabilizationGate(
            parameter_name="Temperature",
            target_value=self.target_temperature,
            tolerance=self.temperature_tolerance_c,
            is_relative=False,
            unit="°C"
        )
        if self.start_time is None:
            self.start_time = datetime.now()

    def add_reading(
        self,
        irradiance: float,
        temperature: float,
        timestamp: Optional[datetime] = None
    ) -> StabilizationResult:
        """Add a new reading and check stabilization status.

        Args:
            irradiance: Measured irradiance (W/m²)
            temperature: Measured temperature (°C)
            timestamp: Reading timestamp (default: now)

        Returns:
            StabilizationResult with current status
        """
        if timestamp is None:
            timestamp = datetime.now()

        reading = StabilizationReading(
            timestamp=timestamp,
            irradiance=irradiance,
            temperature=temperature
        )
        self.readings.append(reading)

        # Remove old readings (keep last 10 minutes)
        cutoff = datetime.now() - timedelta(seconds=600)
        self.readings = [r for r in self.readings if r.timestamp > cutoff]

        return self.check_status()

    def check_status(self) -> StabilizationResult:
        """Check current stabilization status.

        Returns:
            StabilizationResult with detailed status information
        """
        if len(self.readings) < self.min_stable_readings:
            return StabilizationResult(
                is_stable=False,
                irradiance_stable=False,
                temperature_stable=False,
                irradiance_deviation_percent=0.0,
                temperature_deviation_c=0.0,
                status=StabilizationStatus.WAITING,
                message=f"Waiting for readings ({len(self.readings)}/{self.min_stable_readings})",
                stable_duration_s=0.0,
                readings_count=len(self.readings)
            )

        # Check timeout
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.max_wait_time_s:
            return StabilizationResult(
                is_stable=False,
                irradiance_stable=False,
                temperature_stable=False,
                irradiance_deviation_percent=0.0,
                temperature_deviation_c=0.0,
                status=StabilizationStatus.TIMEOUT,
                message=f"Timeout after {elapsed:.1f}s",
                stable_duration_s=0.0,
                readings_count=len(self.readings)
            )

        # Get recent readings for analysis
        recent_readings = self.readings[-self.min_stable_readings:]

        # Check irradiance stability
        irradiances = [r.irradiance for r in recent_readings]
        irr_stable = all(
            self.irradiance_gate.is_within_gate(g)
            for g in irradiances
        )
        irr_deviation = self.irradiance_gate.deviation(np.mean(irradiances))

        # Check temperature stability
        temperatures = [r.temperature for r in recent_readings]
        temp_stable = all(
            self.temperature_gate.is_within_gate(t)
            for t in temperatures
        )
        temp_deviation = self.temperature_gate.deviation(np.mean(temperatures))

        is_stable = irr_stable and temp_stable

        # Track stable duration
        if is_stable:
            if self.stable_since is None:
                self.stable_since = recent_readings[0].timestamp
            stable_duration = (datetime.now() - self.stable_since).total_seconds()
        else:
            self.stable_since = None
            stable_duration = 0.0

        # Check if minimum stabilization time met
        ready = is_stable and stable_duration >= self.min_stabilization_time_s

        if ready:
            status = StabilizationStatus.STABLE
            message = f"Stable for {stable_duration:.1f}s - Ready for measurement"
        elif is_stable:
            status = StabilizationStatus.WAITING
            remaining = self.min_stabilization_time_s - stable_duration
            message = f"Conditions stable, waiting {remaining:.1f}s more"
        else:
            status = StabilizationStatus.UNSTABLE
            issues = []
            if not irr_stable:
                issues.append(f"Irradiance ({irr_deviation:.2f}%)")
            if not temp_stable:
                issues.append(f"Temperature ({temp_deviation:.2f}°C)")
            message = f"Unstable: {', '.join(issues)}"

        return StabilizationResult(
            is_stable=ready,
            irradiance_stable=irr_stable,
            temperature_stable=temp_stable,
            irradiance_deviation_percent=irr_deviation,
            temperature_deviation_c=temp_deviation,
            status=status,
            message=message,
            stable_duration_s=stable_duration,
            readings_count=len(self.readings)
        )

    def reset(self):
        """Reset monitor for new measurement cycle."""
        self.readings.clear()
        self.start_time = datetime.now()
        self.stable_since = None

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics of collected readings.

        Returns:
            Dictionary with min, max, mean, std for irradiance and temperature
        """
        if not self.readings:
            return {}

        irradiances = [r.irradiance for r in self.readings]
        temperatures = [r.temperature for r in self.readings]

        return {
            "irradiance_min": float(np.min(irradiances)),
            "irradiance_max": float(np.max(irradiances)),
            "irradiance_mean": float(np.mean(irradiances)),
            "irradiance_std": float(np.std(irradiances)),
            "temperature_min": float(np.min(temperatures)),
            "temperature_max": float(np.max(temperatures)),
            "temperature_mean": float(np.mean(temperatures)),
            "temperature_std": float(np.std(temperatures)),
            "readings_count": len(self.readings),
            "duration_s": (self.readings[-1].timestamp - self.readings[0].timestamp).total_seconds()
            if len(self.readings) > 1 else 0.0
        }


def check_irradiance_gate(
    irradiance: float,
    target: float = STC.irradiance,
    tolerance_percent: float = STABILIZATION_GATES.irradiance_tolerance_percent
) -> Tuple[bool, float]:
    """Quick check if irradiance is within gate.

    Args:
        irradiance: Measured irradiance (W/m²)
        target: Target irradiance (default: STC)
        tolerance_percent: Tolerance in percent

    Returns:
        (is_within_gate, deviation_percent)
    """
    tolerance = target * (tolerance_percent / 100.0)
    deviation = abs(irradiance - target)
    deviation_percent = (deviation / target) * 100.0

    return (deviation <= tolerance, deviation_percent)


def check_temperature_gate(
    temperature: float,
    target: float = STC.temperature,
    tolerance_c: float = STABILIZATION_GATES.temperature_tolerance_c
) -> Tuple[bool, float]:
    """Quick check if temperature is within gate.

    Args:
        temperature: Measured temperature (°C)
        target: Target temperature (default: STC)
        tolerance_c: Tolerance in °C

    Returns:
        (is_within_gate, deviation_c)
    """
    deviation = abs(temperature - target)
    return (deviation <= tolerance_c, deviation)


def calculate_non_uniformity(values: np.ndarray) -> float:
    """Calculate spatial non-uniformity per IEC 60904-9.

    Non-uniformity = ((max - min) / (max + min)) * 100%

    Args:
        values: Array of irradiance measurements across test area

    Returns:
        Non-uniformity percentage
    """
    if len(values) < 2:
        return 0.0

    max_val = np.max(values)
    min_val = np.min(values)

    if (max_val + min_val) == 0:
        return 0.0

    return ((max_val - min_val) / (max_val + min_val)) * 100.0


def calculate_temporal_instability(values: np.ndarray) -> float:
    """Calculate temporal instability per IEC 60904-9.

    Instability = ((max - min) / (max + min)) * 100%

    Args:
        values: Array of irradiance measurements over time

    Returns:
        Temporal instability percentage
    """
    return calculate_non_uniformity(values)  # Same formula


def validate_measurement_conditions(
    irradiance: float,
    temperature: float,
    target_irradiance: float = STC.irradiance,
    target_temperature: float = STC.temperature,
    irradiance_tolerance_percent: float = 2.0,
    temperature_tolerance_c: float = 2.0
) -> Dict[str, any]:
    """Validate if measurement conditions meet IEC requirements.

    Args:
        irradiance: Measured irradiance (W/m²)
        temperature: Measured temperature (°C)
        target_irradiance: Target irradiance (default: STC)
        target_temperature: Target temperature (default: STC)
        irradiance_tolerance_percent: Irradiance tolerance (%)
        temperature_tolerance_c: Temperature tolerance (°C)

    Returns:
        Dictionary with validation results
    """
    irr_ok, irr_dev = check_irradiance_gate(
        irradiance, target_irradiance, irradiance_tolerance_percent
    )
    temp_ok, temp_dev = check_temperature_gate(
        temperature, target_temperature, temperature_tolerance_c
    )

    overall_valid = irr_ok and temp_ok

    return {
        "valid": overall_valid,
        "irradiance": {
            "value": irradiance,
            "target": target_irradiance,
            "within_tolerance": irr_ok,
            "deviation_percent": irr_dev,
            "tolerance_percent": irradiance_tolerance_percent
        },
        "temperature": {
            "value": temperature,
            "target": target_temperature,
            "within_tolerance": temp_ok,
            "deviation_c": temp_dev,
            "tolerance_c": temperature_tolerance_c
        },
        "message": "Conditions valid" if overall_valid else "Conditions out of tolerance"
    }


class EnvironmentalMonitor:
    """Monitor environmental conditions during extended tests.

    Tracks ambient temperature, humidity, and pressure for
    environmental correction factors.
    """

    def __init__(self):
        """Initialize environmental monitor."""
        self.readings: List[Dict] = []

    def add_reading(
        self,
        ambient_temp_c: float,
        relative_humidity_percent: float,
        pressure_hpa: float = 1013.25,
        timestamp: Optional[datetime] = None
    ):
        """Add environmental reading.

        Args:
            ambient_temp_c: Ambient temperature (°C)
            relative_humidity_percent: Relative humidity (%)
            pressure_hpa: Atmospheric pressure (hPa)
            timestamp: Reading timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.readings.append({
            "timestamp": timestamp,
            "ambient_temperature_c": ambient_temp_c,
            "relative_humidity_percent": relative_humidity_percent,
            "pressure_hpa": pressure_hpa
        })

    def get_average_conditions(self) -> Dict[str, float]:
        """Get average environmental conditions.

        Returns:
            Dictionary with average values
        """
        if not self.readings:
            return {}

        return {
            "ambient_temperature_c": np.mean(
                [r["ambient_temperature_c"] for r in self.readings]
            ),
            "relative_humidity_percent": np.mean(
                [r["relative_humidity_percent"] for r in self.readings]
            ),
            "pressure_hpa": np.mean(
                [r["pressure_hpa"] for r in self.readings]
            ),
            "readings_count": len(self.readings)
        }

    def check_humidity_limits(
        self,
        min_rh: float = 40.0,
        max_rh: float = 75.0
    ) -> Tuple[bool, str]:
        """Check if humidity is within acceptable limits.

        Args:
            min_rh: Minimum acceptable RH (%)
            max_rh: Maximum acceptable RH (%)

        Returns:
            (is_within_limits, message)
        """
        if not self.readings:
            return (False, "No readings available")

        rh_values = [r["relative_humidity_percent"] for r in self.readings]
        avg_rh = np.mean(rh_values)

        if avg_rh < min_rh:
            return (False, f"Humidity too low: {avg_rh:.1f}% < {min_rh}%")
        elif avg_rh > max_rh:
            return (False, f"Humidity too high: {avg_rh:.1f}% > {max_rh}%")

        return (True, f"Humidity OK: {avg_rh:.1f}%")
