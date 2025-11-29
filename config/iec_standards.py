"""IEC Standards Parameters and Constants.

IEC 60904: Photovoltaic devices
IEC 60891: Procedures for temperature and irradiance corrections
IEC 61215: Terrestrial photovoltaic modules - Design qualification
"""

from typing import Dict, List, Any
from dataclasses import dataclass

# ============================================================================
# IEC 60904 - PHOTOVOLTAIC DEVICES
# ============================================================================

@dataclass
class STCConditions:
    """Standard Test Conditions (STC) per IEC 60904-3."""
    temperature: float = 25.0  # °C
    irradiance: float = 1000.0  # W/m²
    air_mass: float = 1.5
    spectrum: str = "AM1.5G"  # Global spectrum

@dataclass
class NOMTConditions:
    """Nominal Operating Module Temperature conditions."""
    irradiance: float = 800.0  # W/m²
    ambient_temp: float = 20.0  # °C
    wind_speed: float = 1.0  # m/s

# ============================================================================
# IEC 60891 - CORRECTION PROCEDURES
# ============================================================================

CORRECTION_PROCEDURES = {
    "procedure_1": {
        "name": "Translation of I-V curve to STC",
        "description": "Corrects for temperature and irradiance using measured parameters",
        "requires": ["alpha", "beta", "Rs", "kappa"],
        "applicable_to": "All PV technologies",
    },
    "procedure_2": {
        "name": "Translation using measured curves",
        "description": "Uses multiple I-V curves at different conditions",
        "requires": ["multiple_iv_curves"],
        "applicable_to": "All PV technologies",
    },
    "procedure_3": {
        "name": "Translation using series resistance",
        "description": "Simplified correction using Rs",
        "requires": ["Rs", "temperature_coefficient"],
        "applicable_to": "Crystalline silicon mainly",
    },
    "procedure_4": {
        "name": "Translation without Rs",
        "description": "Simplified correction without series resistance",
        "requires": ["temperature_coefficient"],
        "applicable_to": "Quick field measurements",
    },
}

# ============================================================================
# IEC 61215 - MODULE QUALIFICATION TESTS
# ============================================================================

QUALIFICATION_TESTS = [
    "Visual Inspection",
    "Maximum Power Determination",
    "Insulation Test",
    "Temperature Coefficients",
    "NOCT Measurement",
    "Low Irradiance Performance",
    "Outdoor Exposure",
    "Hot-Spot Endurance",
    "UV Preconditioning",
    "Thermal Cycling",
    "Humidity-Freeze",
    "Damp Heat",
    "Robustness of Terminations",
    "Wet Leakage Current",
    "Mechanical Load",
    "Hail Impact",
    "Bypass Diode Thermal Test",
]

# ============================================================================
# SPECTRAL MISMATCH CORRECTION (IEC 60904-7)
# ============================================================================

SPECTRAL_RANGE = {
    "wavelength_min": 300,  # nm
    "wavelength_max": 1200,  # nm
    "wavelength_step": 1,  # nm
    "bins": [
        {"range": "300-470", "name": "UV-Blue"},
        {"range": "470-561", "name": "Green"},
        {"range": "561-657", "name": "Yellow-Orange"},
        {"range": "657-772", "name": "Red"},
        {"range": "772-919", "name": "Near-IR 1"},
        {"range": "919-1200", "name": "Near-IR 2"},
    ],
}

# ============================================================================
# UNCERTAINTY BUDGET COMPONENTS (GUM)
# ============================================================================

UNCERTAINTY_SOURCES = [
    "Irradiance sensor calibration",
    "Irradiance non-uniformity",
    "Irradiance temporal instability",
    "Spectral mismatch",
    "Temperature measurement",
    "I-V measurement system",
    "Module temperature non-uniformity",
    "Repeatability",
    "Reproducibility",
]

# ============================================================================
# TEMPERATURE COEFFICIENT MEASUREMENT
# ============================================================================

TEMP_COEFF_REQUIREMENTS = {
    "min_temperature_range": 15,  # °C
    "min_measurements": 3,
    "irradiance_tolerance": 0.02,  # ±2%
    "typical_values": {
        "c-Si": {
            "alpha_isc": 0.0005,  # A/°C per A at STC (0.05%/°C)
            "beta_voc": -0.0030,  # V/°C per V at STC (-0.30%/°C)
            "gamma_pmax": -0.0045,  # W/°C per W at STC (-0.45%/°C)
        },
        "CdTe": {
            "alpha_isc": 0.0004,
            "beta_voc": -0.0025,
            "gamma_pmax": -0.0025,
        },
        "CIGS": {
            "alpha_isc": 0.0003,
            "beta_voc": -0.0036,
            "gamma_pmax": -0.0032,
        },
    },
}

# ============================================================================
# HOTSPOT DETECTION CRITERIA
# ============================================================================

HOTSPOT_CRITERIA = {
    "temperature_threshold": 10.0,  # °C above average
    "power_degradation_threshold": 0.05,  # 5%
    "cell_configurations": ["60-cell", "72-cell", "120-half-cut", "144-half-cut"],
    "detection_methods": ["thermal_imaging", "reverse_bias", "statistical_analysis"],
}

# ============================================================================
# ENERGY RATING (IEC 61853)
# ============================================================================

ENERGY_RATING_CONDITIONS = [
    {"irradiance": 1000, "temperature": 15},
    {"irradiance": 1000, "temperature": 25},
    {"irradiance": 1000, "temperature": 50},
    {"irradiance": 1000, "temperature": 75},
    {"irradiance": 800, "temperature": 25},
    {"irradiance": 600, "temperature": 25},
    {"irradiance": 400, "temperature": 25},
    {"irradiance": 200, "temperature": 25},
]
