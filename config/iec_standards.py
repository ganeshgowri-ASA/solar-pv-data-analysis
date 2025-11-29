"""IEC Standards Parameters and Constants.

IEC 60904: Photovoltaic devices
IEC 60904-9: Solar simulator performance requirements
IEC 60891: Procedures for temperature and irradiance corrections
IEC 61215: Terrestrial photovoltaic modules - Design qualification
IEC 60904-7: Spectral mismatch correction
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# IEC 60904-9 SOLAR SIMULATOR CLASSIFICATION
# ============================================================================

class SimulatorClass(Enum):
    """Solar simulator classification grades per IEC 60904-9:2020."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class SpectralMatchCriteria:
    """Spectral match classification criteria per IEC 60904-9.

    The spectral match is the ratio of measured irradiance to reference
    irradiance in each wavelength band.
    """
    # Class A+: 0.875 to 1.125 (±12.5%)
    a_plus_min: float = 0.875
    a_plus_max: float = 1.125

    # Class A: 0.75 to 1.25 (±25%)
    a_min: float = 0.75
    a_max: float = 1.25

    # Class B: 0.6 to 1.4 (±40%)
    b_min: float = 0.6
    b_max: float = 1.4

    # Class C: 0.4 to 2.0 (±60%/+100%)
    c_min: float = 0.4
    c_max: float = 2.0

    def classify(self, ratio: float) -> SimulatorClass:
        """Classify a spectral match ratio."""
        if self.a_plus_min <= ratio <= self.a_plus_max:
            return SimulatorClass.A_PLUS
        elif self.a_min <= ratio <= self.a_max:
            return SimulatorClass.A
        elif self.b_min <= ratio <= self.b_max:
            return SimulatorClass.B
        elif self.c_min <= ratio <= self.c_max:
            return SimulatorClass.C
        else:
            raise ValueError(f"Spectral match ratio {ratio} is outside Class C limits")


@dataclass(frozen=True)
class NonUniformityCriteria:
    """Spatial non-uniformity of irradiance classification per IEC 60904-9.

    Non-uniformity is calculated as: ((max - min) / (max + min)) * 100%
    """
    # Class A+: ≤1%
    a_plus_max: float = 1.0

    # Class A: ≤2%
    a_max: float = 2.0

    # Class B: ≤5%
    b_max: float = 5.0

    # Class C: ≤10%
    c_max: float = 10.0

    def classify(self, non_uniformity_percent: float) -> SimulatorClass:
        """Classify non-uniformity value."""
        if non_uniformity_percent <= self.a_plus_max:
            return SimulatorClass.A_PLUS
        elif non_uniformity_percent <= self.a_max:
            return SimulatorClass.A
        elif non_uniformity_percent <= self.b_max:
            return SimulatorClass.B
        elif non_uniformity_percent <= self.c_max:
            return SimulatorClass.C
        else:
            raise ValueError(
                f"Non-uniformity {non_uniformity_percent}% exceeds Class C limit of {self.c_max}%"
            )


@dataclass(frozen=True)
class ShortTermInstabilityCriteria:
    """Short-term temporal instability (STI) classification per IEC 60904-9.

    STI is measured during the data acquisition period (typically <1 second for flash).
    Calculated as: ((max - min) / (max + min)) * 100%
    """
    # Class A+: ≤0.25%
    a_plus_max: float = 0.25

    # Class A: ≤0.5%
    a_max: float = 0.5

    # Class B: ≤2%
    b_max: float = 2.0

    # Class C: ≤10%
    c_max: float = 10.0

    def classify(self, sti_percent: float) -> SimulatorClass:
        """Classify STI value."""
        if sti_percent <= self.a_plus_max:
            return SimulatorClass.A_PLUS
        elif sti_percent <= self.a_max:
            return SimulatorClass.A
        elif sti_percent <= self.b_max:
            return SimulatorClass.B
        elif sti_percent <= self.c_max:
            return SimulatorClass.C
        else:
            raise ValueError(
                f"STI {sti_percent}% exceeds Class C limit of {self.c_max}%"
            )


@dataclass(frozen=True)
class LongTermInstabilityCriteria:
    """Long-term temporal instability (LTI) classification per IEC 60904-9.

    LTI is measured over the entire test duration (multiple flashes or continuous).
    Calculated as: ((max - min) / (max + min)) * 100%
    """
    # Class A+: <1%
    a_plus_max: float = 1.0

    # Class A: ≤2%
    a_max: float = 2.0

    # Class B: ≤5%
    b_max: float = 5.0

    # Class C: ≤10%
    c_max: float = 10.0

    def classify(self, lti_percent: float) -> SimulatorClass:
        """Classify LTI value."""
        if lti_percent < self.a_plus_max:
            return SimulatorClass.A_PLUS
        elif lti_percent <= self.a_max:
            return SimulatorClass.A
        elif lti_percent <= self.b_max:
            return SimulatorClass.B
        elif lti_percent <= self.c_max:
            return SimulatorClass.C
        else:
            raise ValueError(
                f"LTI {lti_percent}% exceeds Class C limit of {self.c_max}%"
            )


@dataclass(frozen=True)
class WavelengthBand:
    """Wavelength band for spectral match evaluation per IEC 60904-9."""
    name: str
    wavelength_min_nm: int
    wavelength_max_nm: int
    description: str = ""

    @property
    def range_str(self) -> str:
        """Return range as string."""
        return f"{self.wavelength_min_nm}-{self.wavelength_max_nm}"

    @property
    def center_wavelength(self) -> float:
        """Return center wavelength."""
        return (self.wavelength_min_nm + self.wavelength_max_nm) / 2


# IEC 60904-9 defined wavelength bands for spectral match evaluation
WAVELENGTH_BANDS: Tuple[WavelengthBand, ...] = (
    WavelengthBand(
        name="Band 1",
        wavelength_min_nm=300,
        wavelength_max_nm=470,
        description="UV to Blue"
    ),
    WavelengthBand(
        name="Band 2",
        wavelength_min_nm=470,
        wavelength_max_nm=561,
        description="Blue to Green"
    ),
    WavelengthBand(
        name="Band 3",
        wavelength_min_nm=561,
        wavelength_max_nm=657,
        description="Green to Orange"
    ),
    WavelengthBand(
        name="Band 4",
        wavelength_min_nm=657,
        wavelength_max_nm=772,
        description="Orange to Red"
    ),
    WavelengthBand(
        name="Band 5",
        wavelength_min_nm=772,
        wavelength_max_nm=919,
        description="Red to Near-IR"
    ),
    WavelengthBand(
        name="Band 6",
        wavelength_min_nm=919,
        wavelength_max_nm=1200,
        description="Near-IR"
    ),
)


@dataclass
class IEC60904_9_Classification:
    """Complete solar simulator classification per IEC 60904-9:2020.

    The overall classification is the worst of the four criteria:
    - Spectral match (all 6 wavelength bands)
    - Spatial non-uniformity of irradiance
    - Short-term temporal instability (STI)
    - Long-term temporal instability (LTI)

    Classification is expressed as three letters: e.g., "AAA" or "ABA"
    representing (Spectral, Non-uniformity, Temporal Instability).

    With the 2020 revision, A+ class was introduced.
    """
    spectral_class: SimulatorClass = SimulatorClass.A
    non_uniformity_class: SimulatorClass = SimulatorClass.A
    temporal_class: SimulatorClass = SimulatorClass.A  # Worst of STI and LTI

    # Detailed values
    spectral_ratios: Dict[str, float] = field(default_factory=dict)
    non_uniformity_percent: float = 0.0
    sti_percent: float = 0.0
    lti_percent: float = 0.0

    @property
    def classification_string(self) -> str:
        """Return classification as string (e.g., 'AAA' or 'A+AA')."""
        return f"{self.spectral_class.value}{self.non_uniformity_class.value}{self.temporal_class.value}"

    @property
    def overall_class(self) -> SimulatorClass:
        """Return overall classification (worst of all criteria)."""
        class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]
        classes = [self.spectral_class, self.non_uniformity_class, self.temporal_class]

        worst_idx = max(class_order.index(c) for c in classes)
        return class_order[worst_idx]

    def is_acceptable_for_stc(self) -> bool:
        """Check if simulator is acceptable for STC measurements (Class A or better)."""
        return self.overall_class in (SimulatorClass.A_PLUS, SimulatorClass.A)


# Instantiate criteria objects for easy access
SPECTRAL_MATCH_CRITERIA = SpectralMatchCriteria()
NON_UNIFORMITY_CRITERIA = NonUniformityCriteria()
STI_CRITERIA = ShortTermInstabilityCriteria()
LTI_CRITERIA = LongTermInstabilityCriteria()


# ============================================================================
# IEC 60904 - PHOTOVOLTAIC DEVICES
# ============================================================================

@dataclass(frozen=True)
class STCConditions:
    """Standard Test Conditions (STC) per IEC 60904-3."""
    temperature: float = 25.0  # °C
    irradiance: float = 1000.0  # W/m²
    air_mass: float = 1.5
    spectrum: str = "AM1.5G"  # Global spectrum

    # Tolerances for STC measurement
    temperature_tolerance: float = 2.0  # ±°C
    irradiance_tolerance_percent: float = 2.0  # ±%


@dataclass(frozen=True)
class NOMTConditions:
    """Nominal Operating Module Temperature conditions per IEC 61215."""
    irradiance: float = 800.0  # W/m²
    ambient_temp: float = 20.0  # °C
    wind_speed: float = 1.0  # m/s
    mounting: str = "open_rack"


# Standard conditions instances
STC = STCConditions()
NOMT = NOMTConditions()


# ============================================================================
# IEC 60891 - CORRECTION PROCEDURES
# ============================================================================

@dataclass
class CorrectionProcedureInfo:
    """Information about IEC 60891 correction procedures."""
    procedure_number: int
    name: str
    description: str
    required_parameters: Tuple[str, ...]
    applicable_to: str
    accuracy: str


CORRECTION_PROCEDURES: Dict[int, CorrectionProcedureInfo] = {
    1: CorrectionProcedureInfo(
        procedure_number=1,
        name="Full I-V curve translation",
        description="Corrects entire I-V curve to STC using all temperature coefficients",
        required_parameters=("alpha", "beta", "Rs", "kappa"),
        applicable_to="All PV technologies",
        accuracy="Highest accuracy (±1%)"
    ),
    2: CorrectionProcedureInfo(
        procedure_number=2,
        name="Interpolation from multiple curves",
        description="Uses multiple I-V curves at different conditions for interpolation",
        required_parameters=("multiple_iv_curves",),
        applicable_to="All PV technologies, requires multiple measurements",
        accuracy="High accuracy (±1-2%)"
    ),
    3: CorrectionProcedureInfo(
        procedure_number=3,
        name="Simplified with series resistance",
        description="Simplified correction using Rs, assumes kappa=0",
        required_parameters=("alpha", "beta", "Rs"),
        applicable_to="Crystalline silicon modules",
        accuracy="Good accuracy (±2%)"
    ),
    4: CorrectionProcedureInfo(
        procedure_number=4,
        name="Quick correction without Rs",
        description="Simplest correction for field measurements",
        required_parameters=("alpha", "beta"),
        applicable_to="Quick field measurements",
        accuracy="Moderate accuracy (±3-5%)"
    ),
}


# ============================================================================
# IEC 60904-7 - SPECTRAL MISMATCH CORRECTION
# ============================================================================

@dataclass(frozen=True)
class SpectralMismatchConfig:
    """Configuration for spectral mismatch calculation per IEC 60904-7."""
    wavelength_min_nm: int = 300
    wavelength_max_nm: int = 1200
    wavelength_step_nm: int = 1

    # Reference spectrum
    reference_spectrum: str = "AM1.5G"

    # Typical M factor range for crystalline silicon
    m_factor_typical_min: float = 0.98
    m_factor_typical_max: float = 1.02


SPECTRAL_MISMATCH_CONFIG = SpectralMismatchConfig()


# ============================================================================
# IEC 61215 - MODULE QUALIFICATION TESTS
# ============================================================================

QUALIFICATION_TESTS: Tuple[str, ...] = (
    "10.1 Visual Inspection",
    "10.2 Maximum Power Determination",
    "10.3 Insulation Test",
    "10.4 Measurement of Temperature Coefficients",
    "10.5 Measurement of NOCT",
    "10.6 Performance at Low Irradiance",
    "10.7 Outdoor Exposure Test",
    "10.8 Hot-Spot Endurance Test",
    "10.9 UV Preconditioning Test",
    "10.10 Thermal Cycling Test",
    "10.11 Humidity-Freeze Test",
    "10.12 Damp Heat Test",
    "10.13 Robustness of Terminations Test",
    "10.14 Wet Leakage Current Test",
    "10.15 Mechanical Load Test",
    "10.16 Hail Impact Test",
    "10.17 Bypass Diode Thermal Test",
    "10.18 Reverse Current Overload Test",
    "10.19 Module Breakage Test",
    "10.20 PID Test (Potential Induced Degradation)",
)


# ============================================================================
# TEMPERATURE COEFFICIENT MEASUREMENT (IEC 60904-10)
# ============================================================================

@dataclass(frozen=True)
class TempCoeffRequirements:
    """Requirements for temperature coefficient measurement per IEC 60904-10."""
    min_temperature_range_c: float = 15.0  # Minimum span in °C
    min_measurement_points: int = 4
    irradiance_tolerance_percent: float = 1.0  # ±1%
    temperature_stability_c: float = 1.0  # ±1°C during measurement

    # Regression requirements
    min_r_squared: float = 0.99


@dataclass
class TypicalTempCoefficients:
    """Typical temperature coefficients by technology."""
    technology: str
    alpha_isc_percent_per_c: float  # %/°C
    beta_voc_percent_per_c: float  # %/°C (negative)
    gamma_pmax_percent_per_c: float  # %/°C (negative)

    @property
    def alpha_isc_absolute(self) -> float:
        """Return alpha as fraction per °C."""
        return self.alpha_isc_percent_per_c / 100.0

    @property
    def beta_voc_absolute(self) -> float:
        """Return beta as fraction per °C."""
        return self.beta_voc_percent_per_c / 100.0

    @property
    def gamma_pmax_absolute(self) -> float:
        """Return gamma as fraction per °C."""
        return self.gamma_pmax_percent_per_c / 100.0


TYPICAL_TEMP_COEFFICIENTS: Dict[str, TypicalTempCoefficients] = {
    "c-Si": TypicalTempCoefficients(
        technology="Crystalline Silicon (mono/poly)",
        alpha_isc_percent_per_c=0.05,
        beta_voc_percent_per_c=-0.30,
        gamma_pmax_percent_per_c=-0.40
    ),
    "HJT": TypicalTempCoefficients(
        technology="Heterojunction (HJT/HIT)",
        alpha_isc_percent_per_c=0.03,
        beta_voc_percent_per_c=-0.25,
        gamma_pmax_percent_per_c=-0.26
    ),
    "TOPCon": TypicalTempCoefficients(
        technology="Tunnel Oxide Passivated Contact",
        alpha_isc_percent_per_c=0.04,
        beta_voc_percent_per_c=-0.27,
        gamma_pmax_percent_per_c=-0.32
    ),
    "CdTe": TypicalTempCoefficients(
        technology="Cadmium Telluride",
        alpha_isc_percent_per_c=0.04,
        beta_voc_percent_per_c=-0.25,
        gamma_pmax_percent_per_c=-0.25
    ),
    "CIGS": TypicalTempCoefficients(
        technology="Copper Indium Gallium Selenide",
        alpha_isc_percent_per_c=0.01,
        beta_voc_percent_per_c=-0.35,
        gamma_pmax_percent_per_c=-0.36
    ),
    "a-Si": TypicalTempCoefficients(
        technology="Amorphous Silicon",
        alpha_isc_percent_per_c=0.07,
        beta_voc_percent_per_c=-0.35,
        gamma_pmax_percent_per_c=-0.20
    ),
    "Perovskite": TypicalTempCoefficients(
        technology="Perovskite",
        alpha_isc_percent_per_c=0.02,
        beta_voc_percent_per_c=-0.20,
        gamma_pmax_percent_per_c=-0.15
    ),
}

TEMP_COEFF_REQUIREMENTS = TempCoeffRequirements()


# ============================================================================
# MEASUREMENT STABILIZATION GATES (IEC 60904-1)
# ============================================================================

@dataclass(frozen=True)
class StabilizationGates:
    """Stabilization gate criteria for measurements per IEC 60904-1."""
    # Irradiance stability requirement
    irradiance_tolerance_percent: float = 1.0  # ±1%

    # Temperature stability requirement
    temperature_tolerance_c: float = 1.0  # ±1°C

    # Minimum stabilization time (seconds)
    min_stabilization_time_s: float = 60.0

    # Number of consecutive readings within tolerance
    min_stable_readings: int = 5

    def is_irradiance_stable(
        self,
        irradiance_values: List[float],
        target_irradiance: float
    ) -> bool:
        """Check if irradiance is within tolerance."""
        if not irradiance_values:
            return False

        tolerance = target_irradiance * (self.irradiance_tolerance_percent / 100.0)
        return all(
            abs(g - target_irradiance) <= tolerance
            for g in irradiance_values[-self.min_stable_readings:]
        )

    def is_temperature_stable(
        self,
        temperature_values: List[float],
        target_temperature: float
    ) -> bool:
        """Check if temperature is within tolerance."""
        if not temperature_values:
            return False

        return all(
            abs(t - target_temperature) <= self.temperature_tolerance_c
            for t in temperature_values[-self.min_stable_readings:]
        )


STABILIZATION_GATES = StabilizationGates()


# ============================================================================
# UNCERTAINTY BUDGET COMPONENTS (GUM)
# ============================================================================

@dataclass(frozen=True)
class UncertaintyComponent:
    """Uncertainty component for GUM analysis."""
    name: str
    symbol: str
    typical_value_percent: float
    distribution: str  # normal, rectangular, triangular
    divisor: float  # For standard uncertainty calculation
    description: str = ""


UNCERTAINTY_COMPONENTS: Tuple[UncertaintyComponent, ...] = (
    UncertaintyComponent(
        name="Reference cell calibration",
        symbol="u_cal",
        typical_value_percent=1.0,
        distribution="normal",
        divisor=2.0,
        description="Calibration uncertainty of reference cell (k=2)"
    ),
    UncertaintyComponent(
        name="Irradiance non-uniformity",
        symbol="u_nonunif",
        typical_value_percent=1.0,
        distribution="rectangular",
        divisor=1.732,  # sqrt(3)
        description="Spatial non-uniformity of irradiance"
    ),
    UncertaintyComponent(
        name="Short-term instability",
        symbol="u_sti",
        typical_value_percent=0.5,
        distribution="rectangular",
        divisor=1.732,
        description="Temporal instability during measurement"
    ),
    UncertaintyComponent(
        name="Long-term instability",
        symbol="u_lti",
        typical_value_percent=1.0,
        distribution="rectangular",
        divisor=1.732,
        description="Temporal instability between measurements"
    ),
    UncertaintyComponent(
        name="Spectral mismatch",
        symbol="u_mm",
        typical_value_percent=1.0,
        distribution="rectangular",
        divisor=1.732,
        description="Spectral mismatch correction uncertainty"
    ),
    UncertaintyComponent(
        name="Temperature measurement",
        symbol="u_temp",
        typical_value_percent=0.5,
        distribution="rectangular",
        divisor=1.732,
        description="Module temperature measurement uncertainty"
    ),
    UncertaintyComponent(
        name="Temperature correction",
        symbol="u_tcorr",
        typical_value_percent=0.3,
        distribution="normal",
        divisor=1.0,
        description="Uncertainty from temperature coefficient"
    ),
    UncertaintyComponent(
        name="I-V measurement system",
        symbol="u_iv",
        typical_value_percent=0.3,
        distribution="normal",
        divisor=1.0,
        description="Electronic measurement uncertainty"
    ),
    UncertaintyComponent(
        name="Data acquisition",
        symbol="u_daq",
        typical_value_percent=0.1,
        distribution="rectangular",
        divisor=1.732,
        description="DAQ system resolution and accuracy"
    ),
    UncertaintyComponent(
        name="Repeatability",
        symbol="u_rep",
        typical_value_percent=0.3,
        distribution="normal",
        divisor=1.0,
        description="Type A uncertainty from repeated measurements"
    ),
)


# ============================================================================
# HOTSPOT DETECTION CRITERIA
# ============================================================================

@dataclass(frozen=True)
class HotspotCriteria:
    """Hotspot detection criteria per IEC 61215-2."""
    temperature_threshold_c: float = 10.0  # °C above average
    power_degradation_threshold_percent: float = 5.0
    max_allowable_temp_c: float = 20.0  # °C above Tc at Pmax


HOTSPOT_CRITERIA = HotspotCriteria()


# ============================================================================
# ENERGY RATING CONDITIONS (IEC 61853)
# ============================================================================

@dataclass(frozen=True)
class EnergyRatingCondition:
    """Single condition point for IEC 61853 energy rating."""
    irradiance: int  # W/m²
    temperature: int  # °C


ENERGY_RATING_MATRIX: Tuple[EnergyRatingCondition, ...] = (
    # Full sun conditions
    EnergyRatingCondition(1100, 25),
    EnergyRatingCondition(1100, 50),
    EnergyRatingCondition(1100, 75),
    EnergyRatingCondition(1000, 15),
    EnergyRatingCondition(1000, 25),  # STC
    EnergyRatingCondition(1000, 50),
    EnergyRatingCondition(1000, 75),
    # Medium irradiance
    EnergyRatingCondition(800, 25),
    EnergyRatingCondition(800, 50),
    EnergyRatingCondition(600, 25),
    EnergyRatingCondition(600, 50),
    # Low irradiance
    EnergyRatingCondition(400, 25),
    EnergyRatingCondition(200, 25),
    EnergyRatingCondition(100, 25),
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_wavelength_band_index(wavelength_nm: float) -> Optional[int]:
    """Get the index of the wavelength band containing the given wavelength."""
    for i, band in enumerate(WAVELENGTH_BANDS):
        if band.wavelength_min_nm <= wavelength_nm < band.wavelength_max_nm:
            return i
    return None


def get_typical_temp_coefficient(
    technology: str,
    parameter: str
) -> Optional[float]:
    """Get typical temperature coefficient for a technology.

    Args:
        technology: PV technology (e.g., 'c-Si', 'HJT', 'CdTe')
        parameter: 'alpha', 'beta', or 'gamma'

    Returns:
        Temperature coefficient as fraction per °C (e.g., -0.004 for -0.4%/°C)
    """
    if technology not in TYPICAL_TEMP_COEFFICIENTS:
        return None

    coeff = TYPICAL_TEMP_COEFFICIENTS[technology]

    if parameter == 'alpha':
        return coeff.alpha_isc_absolute
    elif parameter == 'beta':
        return coeff.beta_voc_absolute
    elif parameter == 'gamma':
        return coeff.gamma_pmax_absolute

    return None


def classify_simulator(
    spectral_ratios: Dict[str, float],
    non_uniformity_percent: float,
    sti_percent: float,
    lti_percent: float
) -> IEC60904_9_Classification:
    """Classify a solar simulator according to IEC 60904-9.

    Args:
        spectral_ratios: Dict mapping band names to spectral match ratios
        non_uniformity_percent: Spatial non-uniformity percentage
        sti_percent: Short-term instability percentage
        lti_percent: Long-term instability percentage

    Returns:
        IEC60904_9_Classification object with full classification details
    """
    # Classify spectral match (worst band determines class)
    spectral_classes = []
    for band_name, ratio in spectral_ratios.items():
        try:
            spectral_classes.append(SPECTRAL_MATCH_CRITERIA.classify(ratio))
        except ValueError:
            spectral_classes.append(SimulatorClass.C)

    class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]
    spectral_class = class_order[max(class_order.index(c) for c in spectral_classes)] if spectral_classes else SimulatorClass.C

    # Classify non-uniformity
    non_uniformity_class = NON_UNIFORMITY_CRITERIA.classify(non_uniformity_percent)

    # Classify temporal stability (worst of STI and LTI)
    sti_class = STI_CRITERIA.classify(sti_percent)
    lti_class = LTI_CRITERIA.classify(lti_percent)
    temporal_class = class_order[max(class_order.index(sti_class), class_order.index(lti_class))]

    return IEC60904_9_Classification(
        spectral_class=spectral_class,
        non_uniformity_class=non_uniformity_class,
        temporal_class=temporal_class,
        spectral_ratios=spectral_ratios,
        non_uniformity_percent=non_uniformity_percent,
        sti_percent=sti_percent,
        lti_percent=lti_percent
    )
