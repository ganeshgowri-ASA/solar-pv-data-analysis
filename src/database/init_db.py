"""Database initialization and protocol seeding.

Provides:
- Database engine setup with PostgreSQL
- get_db() context manager for session handling
- init_database() for table creation
- seed_protocols() for populating all 54 IEC standard protocols
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from .schema import Base, Protocol, ProtocolCategory


# Database URL from environment variable or default for local dev
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/solar_pv'
)

# Handle Railway's postgres:// vs postgresql:// prefix
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create engine with connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Provide a transactional scope around database operations.

    Usage:
        with get_db() as db:
            protocols = db.query(Protocol).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_all_protocols() -> List[Dict[str, Any]]:
    """Return all 54 IEC standard protocols for seeding.

    Protocols cover:
    - IEC 60904-1: I-V measurements (6 protocols)
    - IEC 60891: STC corrections (4 protocols)
    - IEC 60904-9: Simulator classification (4 protocols)
    - IEC 60904-7: Spectral mismatch (2 protocols)
    - IEC 60904-10: Temperature coefficients (3 protocols)
    - IEC 61215: Module qualification (20 protocols)
    - IEC 61853: Energy rating (6 protocols)
    - IEC TS 60904-1-2: Bifaciality (3 protocols)
    - IAM Analysis (3 protocols)
    - Hotspot Detection (3 protocols)
    Total: 54 protocols
    """
    protocols = []

    # =========================================================================
    # IEC 60904-1: I-V CURVE MEASUREMENTS (6 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60904-1-001",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "7",
            "name": "I-V Curve Measurement - Standard",
            "description": "Standard I-V characteristic measurement of photovoltaic devices under natural or simulated sunlight",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature", "voltage_array", "current_array"],
            "optional_parameters": ["spectral_irradiance", "reference_cell_id"],
            "default_values": {"irradiance": 1000, "temperature": 25},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 100.0,
            "irradiance_range_max": 1200.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Class AAA solar simulator", "I-V tracer", "Reference cell"],
            "procedure_steps": [
                "Stabilize module temperature to target ±1°C",
                "Verify irradiance level with reference cell",
                "Perform I-V sweep from Isc to Voc",
                "Record voltage, current, temperature, irradiance",
                "Extract key parameters (Isc, Voc, Pmax, FF)"
            ],
            "pass_criteria": {"temperature_stability": 1.0, "irradiance_stability": 1.0},
            "display_order": 1,
        },
        {
            "protocol_id": "IEC60904-1-002",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "7.2",
            "name": "I-V Curve - Multi-flash Method",
            "description": "I-V measurement using multiple flash pulses for complete curve acquisition",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature", "flash_duration_ms"],
            "optional_parameters": ["flash_count", "delay_between_flashes"],
            "default_values": {"flash_duration_ms": 10, "flash_count": 200},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 1.5,
            "equipment_requirements": ["Flash solar simulator", "High-speed I-V tracer"],
            "procedure_steps": [
                "Configure flash parameters",
                "Position module in test area",
                "Execute multi-flash I-V sweep",
                "Reconstruct complete I-V curve",
                "Validate flash stability"
            ],
            "pass_criteria": {"flash_stability": 0.5},
            "display_order": 2,
        },
        {
            "protocol_id": "IEC60904-1-003",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "7.3",
            "name": "I-V Curve - Continuous Illumination",
            "description": "I-V measurement under steady-state continuous illumination",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature", "sweep_time_ms"],
            "optional_parameters": ["sweep_direction"],
            "default_values": {"sweep_time_ms": 100, "sweep_direction": "forward"},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 100.0,
            "irradiance_range_max": 1200.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Continuous solar simulator", "I-V tracer"],
            "procedure_steps": [
                "Stabilize continuous light source",
                "Wait for module thermal equilibrium",
                "Execute voltage sweep",
                "Record data points",
                "Verify light stability during sweep"
            ],
            "pass_criteria": {"light_stability": 2.0},
            "display_order": 3,
        },
        {
            "protocol_id": "IEC60904-1-004",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "8",
            "name": "I-V Curve - Low Irradiance",
            "description": "I-V measurement at low irradiance levels (200 W/m²)",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature"],
            "optional_parameters": ["reference_cell_low_light"],
            "default_values": {"irradiance": 200, "temperature": 25},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": 100.0,
            "irradiance_range_max": 400.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["Low-light calibrated simulator", "Sensitive I-V tracer"],
            "procedure_steps": [
                "Configure low irradiance level",
                "Verify uniformity at low light",
                "Stabilize temperature",
                "Perform I-V sweep",
                "Record low-light performance"
            ],
            "pass_criteria": {"uniformity_at_low_light": 3.0},
            "display_order": 4,
        },
        {
            "protocol_id": "IEC60904-1-005",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "9",
            "name": "Isc Measurement - Direct Method",
            "description": "Direct measurement of short-circuit current",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature"],
            "optional_parameters": [],
            "default_values": {"irradiance": 1000, "temperature": 25},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Solar simulator", "Low-impedance current meter"],
            "procedure_steps": [
                "Short-circuit device terminals",
                "Illuminate with calibrated light",
                "Measure steady-state current",
                "Record temperature and irradiance"
            ],
            "pass_criteria": {"current_stability": 0.5},
            "display_order": 5,
        },
        {
            "protocol_id": "IEC60904-1-006",
            "standard_reference": "IEC 60904-1",
            "standard_version": "2020",
            "method_clause_no": "10",
            "name": "Voc Measurement - Direct Method",
            "description": "Direct measurement of open-circuit voltage",
            "category": ProtocolCategory.IV_MEASUREMENT,
            "test_type": "I-V Curve",
            "required_parameters": ["irradiance", "temperature"],
            "optional_parameters": [],
            "default_values": {"irradiance": 1000, "temperature": 25},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A",
            "typical_uncertainty_percent": 0.5,
            "equipment_requirements": ["Solar simulator", "High-impedance voltmeter"],
            "procedure_steps": [
                "Open-circuit device terminals",
                "Illuminate with calibrated light",
                "Wait for voltage stabilization",
                "Measure steady-state voltage"
            ],
            "pass_criteria": {"voltage_stability": 0.1},
            "display_order": 6,
        },
    ])

    # =========================================================================
    # IEC 60891: STC CORRECTION PROCEDURES (4 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60891-001",
            "standard_reference": "IEC 60891",
            "standard_version": "2021",
            "method_clause_no": "4.3",
            "name": "STC Correction - Procedure 1",
            "description": "Full I-V curve translation to STC using all temperature coefficients (alpha, beta, Rs, kappa)",
            "category": ProtocolCategory.STC_CORRECTION,
            "test_type": "STC Correction",
            "required_parameters": ["alpha", "beta", "Rs", "kappa"],
            "optional_parameters": ["irradiance_measured", "temperature_measured"],
            "default_values": {"alpha": 0.0005, "beta": -0.003, "Rs": 0.3, "kappa": 0.0},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 700.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±1%",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Calibrated I-V data", "Temperature coefficients"],
            "procedure_steps": [
                "Measure I-V curve at actual conditions",
                "Determine temperature coefficients",
                "Calculate irradiance correction",
                "Apply voltage translation",
                "Apply current translation",
                "Reconstruct STC I-V curve"
            ],
            "pass_criteria": {"correction_accuracy": 1.0},
            "display_order": 10,
        },
        {
            "protocol_id": "IEC60891-002",
            "standard_reference": "IEC 60891",
            "standard_version": "2021",
            "method_clause_no": "4.4",
            "name": "STC Correction - Procedure 2",
            "description": "Interpolation from multiple I-V curves at different conditions",
            "category": ProtocolCategory.STC_CORRECTION,
            "test_type": "STC Correction",
            "required_parameters": ["multiple_iv_curves", "irradiance_levels", "temperature_levels"],
            "optional_parameters": [],
            "default_values": {},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 200.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±1-2%",
            "typical_uncertainty_percent": 1.5,
            "equipment_requirements": ["Multiple I-V curves", "Variable irradiance source"],
            "procedure_steps": [
                "Measure I-V curves at minimum 4 conditions",
                "Create interpolation matrix",
                "Interpolate to STC point",
                "Validate with boundary conditions"
            ],
            "pass_criteria": {"interpolation_residual": 0.5},
            "display_order": 11,
        },
        {
            "protocol_id": "IEC60891-003",
            "standard_reference": "IEC 60891",
            "standard_version": "2021",
            "method_clause_no": "4.5",
            "name": "STC Correction - Procedure 3",
            "description": "Simplified correction using Rs, assumes kappa=0 (crystalline silicon)",
            "category": ProtocolCategory.STC_CORRECTION,
            "test_type": "STC Correction",
            "required_parameters": ["alpha", "beta", "Rs"],
            "optional_parameters": [],
            "default_values": {"alpha": 0.0005, "beta": -0.003, "Rs": 0.3},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["I-V data", "Temperature coefficients", "Series resistance"],
            "procedure_steps": [
                "Measure I-V curve",
                "Apply simplified temperature correction",
                "Apply irradiance correction",
                "Calculate STC parameters"
            ],
            "pass_criteria": {"correction_accuracy": 2.0},
            "display_order": 12,
        },
        {
            "protocol_id": "IEC60891-004",
            "standard_reference": "IEC 60891",
            "standard_version": "2021",
            "method_clause_no": "4.6",
            "name": "STC Correction - Procedure 4",
            "description": "Quick field correction without series resistance (alpha, beta only)",
            "category": ProtocolCategory.STC_CORRECTION,
            "test_type": "STC Correction",
            "required_parameters": ["alpha", "beta"],
            "optional_parameters": [],
            "default_values": {"alpha": 0.0005, "beta": -0.003},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±3-5%",
            "typical_uncertainty_percent": 4.0,
            "equipment_requirements": ["I-V data", "Basic temperature coefficients"],
            "procedure_steps": [
                "Measure I-V curve in field",
                "Apply linear temperature correction",
                "Apply irradiance ratio correction",
                "Report with increased uncertainty"
            ],
            "pass_criteria": {"correction_accuracy": 5.0},
            "display_order": 13,
        },
    ])

    # =========================================================================
    # IEC 60904-9: SIMULATOR CLASSIFICATION (4 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60904-9-001",
            "standard_reference": "IEC 60904-9",
            "standard_version": "2020",
            "method_clause_no": "5",
            "name": "Spectral Match Classification",
            "description": "Classification of solar simulator spectral match across 6 wavelength bands",
            "category": ProtocolCategory.SIMULATOR_CLASSIFICATION,
            "test_type": "Simulator Classification",
            "required_parameters": ["spectral_irradiance", "wavelength_bands"],
            "optional_parameters": ["reference_spectrum"],
            "default_values": {"reference_spectrum": "AM1.5G"},
            "applicable_technologies": ["All"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A+/A/B/C",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Spectroradiometer", "Reference spectrum data"],
            "procedure_steps": [
                "Measure spectral irradiance 300-1200nm",
                "Integrate over 6 wavelength bands",
                "Calculate ratios to AM1.5G reference",
                "Classify each band (A+, A, B, C)",
                "Report worst-case classification"
            ],
            "pass_criteria": {"class_a_range": [0.75, 1.25], "class_a_plus_range": [0.875, 1.125]},
            "display_order": 20,
        },
        {
            "protocol_id": "IEC60904-9-002",
            "standard_reference": "IEC 60904-9",
            "standard_version": "2020",
            "method_clause_no": "6",
            "name": "Spatial Non-Uniformity Classification",
            "description": "Classification of irradiance spatial uniformity across test area",
            "category": ProtocolCategory.SIMULATOR_CLASSIFICATION,
            "test_type": "Simulator Classification",
            "required_parameters": ["irradiance_map", "measurement_grid"],
            "optional_parameters": ["test_area_dimensions"],
            "default_values": {"measurement_grid": "8x8"},
            "applicable_technologies": ["All"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A+/A/B/C",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Reference cell", "XY positioning system"],
            "procedure_steps": [
                "Define test area and grid",
                "Measure irradiance at each grid point",
                "Calculate non-uniformity: (max-min)/(max+min)*100",
                "Classify: A+ ≤1%, A ≤2%, B ≤5%, C ≤10%"
            ],
            "pass_criteria": {"class_a_max": 2.0, "class_a_plus_max": 1.0},
            "display_order": 21,
        },
        {
            "protocol_id": "IEC60904-9-003",
            "standard_reference": "IEC 60904-9",
            "standard_version": "2020",
            "method_clause_no": "7.1",
            "name": "Short-Term Instability (STI) Classification",
            "description": "Classification of temporal instability during data acquisition",
            "category": ProtocolCategory.SIMULATOR_CLASSIFICATION,
            "test_type": "Simulator Classification",
            "required_parameters": ["irradiance_time_series", "acquisition_time_ms"],
            "optional_parameters": ["sampling_rate_hz"],
            "default_values": {"acquisition_time_ms": 10, "sampling_rate_hz": 10000},
            "applicable_technologies": ["All"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A+/A/B/C",
            "typical_uncertainty_percent": 0.5,
            "equipment_requirements": ["High-speed irradiance sensor", "Fast DAQ"],
            "procedure_steps": [
                "Configure high-speed acquisition",
                "Capture irradiance during I-V sweep time",
                "Calculate STI: (max-min)/(max+min)*100",
                "Classify: A+ ≤0.25%, A ≤0.5%, B ≤2%, C ≤10%"
            ],
            "pass_criteria": {"class_a_max": 0.5, "class_a_plus_max": 0.25},
            "display_order": 22,
        },
        {
            "protocol_id": "IEC60904-9-004",
            "standard_reference": "IEC 60904-9",
            "standard_version": "2020",
            "method_clause_no": "7.2",
            "name": "Long-Term Instability (LTI) Classification",
            "description": "Classification of temporal instability over test duration",
            "category": ProtocolCategory.SIMULATOR_CLASSIFICATION,
            "test_type": "Simulator Classification",
            "required_parameters": ["irradiance_time_series", "test_duration_min"],
            "optional_parameters": ["measurement_interval_s"],
            "default_values": {"test_duration_min": 60, "measurement_interval_s": 1},
            "applicable_technologies": ["All"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "A+/A/B/C",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Reference cell", "Continuous monitoring system"],
            "procedure_steps": [
                "Monitor irradiance over test period",
                "Record at regular intervals",
                "Calculate LTI: (max-min)/(max+min)*100",
                "Classify: A+ <1%, A ≤2%, B ≤5%, C ≤10%"
            ],
            "pass_criteria": {"class_a_max": 2.0, "class_a_plus_max": 1.0},
            "display_order": 23,
        },
    ])

    # =========================================================================
    # IEC 60904-7: SPECTRAL MISMATCH (2 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60904-7-001",
            "standard_reference": "IEC 60904-7",
            "standard_version": "2019",
            "method_clause_no": "5",
            "name": "Spectral Mismatch Correction",
            "description": "Calculate spectral mismatch correction factor M",
            "category": ProtocolCategory.SPECTRAL_MISMATCH,
            "test_type": "Spectral Mismatch",
            "required_parameters": ["test_spectrum", "reference_spectrum", "dut_sr", "reference_sr"],
            "optional_parameters": ["wavelength_range"],
            "default_values": {"wavelength_range": [300, 1200]},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±1%",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Spectroradiometer", "SR measurement system"],
            "procedure_steps": [
                "Measure test source spectrum E_test(λ)",
                "Obtain reference spectrum E_ref(λ) - AM1.5G",
                "Measure DUT spectral response SR_DUT(λ)",
                "Use reference cell SR SR_ref(λ)",
                "Calculate M factor by integration"
            ],
            "pass_criteria": {"m_factor_range": [0.97, 1.03]},
            "display_order": 30,
        },
        {
            "protocol_id": "IEC60904-7-002",
            "standard_reference": "IEC 60904-7",
            "standard_version": "2019",
            "method_clause_no": "6",
            "name": "Spectral Response Measurement",
            "description": "Measurement of device spectral response for mismatch calculation",
            "category": ProtocolCategory.SPECTRAL_MISMATCH,
            "test_type": "Spectral Mismatch",
            "required_parameters": ["wavelengths", "responsivity"],
            "optional_parameters": ["bias_light", "temperature"],
            "default_values": {"temperature": 25},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Monochromator", "Calibrated detector", "Bias light source"],
            "procedure_steps": [
                "Calibrate monochromator output",
                "Apply bias light at 1-sun equivalent",
                "Scan wavelengths 300-1200nm",
                "Measure current response at each wavelength",
                "Calculate spectral responsivity A/W"
            ],
            "pass_criteria": {"wavelength_coverage": [300, 1200]},
            "display_order": 31,
        },
    ])

    # =========================================================================
    # IEC 60904-10: TEMPERATURE COEFFICIENTS (3 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60904-10-001",
            "standard_reference": "IEC 60904-10",
            "standard_version": "2020",
            "method_clause_no": "5",
            "name": "Temperature Coefficient - Isc (alpha)",
            "description": "Measurement of short-circuit current temperature coefficient",
            "category": ProtocolCategory.TEMPERATURE_COEFFICIENT,
            "test_type": "Temperature Coefficient",
            "required_parameters": ["temperature_points", "isc_values"],
            "optional_parameters": ["irradiance"],
            "default_values": {"irradiance": 1000},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 900.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±5%",
            "typical_uncertainty_percent": 5.0,
            "equipment_requirements": ["Temperature-controlled chamber", "Solar simulator"],
            "procedure_steps": [
                "Stabilize at minimum 4 temperatures",
                "Temperature span ≥15°C",
                "Measure Isc at each temperature",
                "Perform linear regression",
                "Calculate alpha (%/°C)"
            ],
            "pass_criteria": {"r_squared_min": 0.99, "temp_span_min": 15},
            "display_order": 40,
        },
        {
            "protocol_id": "IEC60904-10-002",
            "standard_reference": "IEC 60904-10",
            "standard_version": "2020",
            "method_clause_no": "6",
            "name": "Temperature Coefficient - Voc (beta)",
            "description": "Measurement of open-circuit voltage temperature coefficient",
            "category": ProtocolCategory.TEMPERATURE_COEFFICIENT,
            "test_type": "Temperature Coefficient",
            "required_parameters": ["temperature_points", "voc_values"],
            "optional_parameters": ["irradiance"],
            "default_values": {"irradiance": 1000},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 900.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±3%",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["Temperature-controlled chamber", "Solar simulator"],
            "procedure_steps": [
                "Stabilize at minimum 4 temperatures",
                "Temperature span ≥15°C",
                "Measure Voc at each temperature",
                "Perform linear regression",
                "Calculate beta (mV/°C)"
            ],
            "pass_criteria": {"r_squared_min": 0.99, "temp_span_min": 15},
            "display_order": 41,
        },
        {
            "protocol_id": "IEC60904-10-003",
            "standard_reference": "IEC 60904-10",
            "standard_version": "2020",
            "method_clause_no": "7",
            "name": "Temperature Coefficient - Pmax (gamma)",
            "description": "Measurement of maximum power temperature coefficient",
            "category": ProtocolCategory.TEMPERATURE_COEFFICIENT,
            "test_type": "Temperature Coefficient",
            "required_parameters": ["temperature_points", "pmax_values"],
            "optional_parameters": ["irradiance"],
            "default_values": {"irradiance": 1000},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 900.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±3%",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["Temperature-controlled chamber", "Solar simulator", "I-V tracer"],
            "procedure_steps": [
                "Stabilize at minimum 4 temperatures",
                "Temperature span ≥15°C",
                "Measure full I-V curve at each temperature",
                "Extract Pmax values",
                "Perform linear regression",
                "Calculate gamma (%/°C)"
            ],
            "pass_criteria": {"r_squared_min": 0.99, "temp_span_min": 15},
            "display_order": 42,
        },
    ])

    # =========================================================================
    # IEC 61215: MODULE QUALIFICATION TESTS (20 protocols)
    # =========================================================================
    qualification_tests = [
        ("10.1", "Visual Inspection", "Visual examination for defects and workmanship"),
        ("10.2", "Maximum Power Determination", "Initial power measurement at STC"),
        ("10.3", "Insulation Test", "Electrical insulation resistance and dielectric withstand"),
        ("10.4", "Measurement of Temperature Coefficients", "Determination of alpha, beta, gamma"),
        ("10.5", "Measurement of NMOT", "Nominal module operating temperature measurement"),
        ("10.6", "Performance at Low Irradiance", "Power measurement at 200 W/m²"),
        ("10.7", "Outdoor Exposure Test", "60 kWh/m² outdoor exposure"),
        ("10.8", "Hot-Spot Endurance Test", "Cell shading test for hot-spot resistance"),
        ("10.9", "UV Preconditioning Test", "15 kWh/m² UV exposure (280-400nm)"),
        ("10.10", "Thermal Cycling Test", "200 cycles from -40°C to +85°C"),
        ("10.11", "Humidity-Freeze Test", "10 cycles with 85% RH"),
        ("10.12", "Damp Heat Test", "1000 hours at 85°C/85% RH"),
        ("10.13", "Robustness of Terminations", "Junction box and cable pull tests"),
        ("10.14", "Wet Leakage Current Test", "Insulation test under wet conditions"),
        ("10.15", "Mechanical Load Test", "2400 Pa uniform load test"),
        ("10.16", "Hail Impact Test", "25mm ice ball at 23 m/s"),
        ("10.17", "Bypass Diode Thermal Test", "Diode operation at 75°C for 1 hour"),
        ("10.18", "Reverse Current Overload", "1.35x Isc reverse current test"),
        ("10.19", "Module Breakage Test", "Mechanical impact resistance"),
        ("10.20", "PID Test", "Potential induced degradation 96 hours at 85°C/85%RH"),
    ]

    for i, (clause, name, desc) in enumerate(qualification_tests):
        protocols.append({
            "protocol_id": f"IEC61215-{clause.replace('.', '')}",
            "standard_reference": "IEC 61215",
            "standard_version": "2021",
            "method_clause_no": clause,
            "name": f"Module Qualification - {name}",
            "description": desc,
            "category": ProtocolCategory.MODULE_QUALIFICATION,
            "test_type": "Module Qualification",
            "required_parameters": ["sample_id", "test_date"],
            "optional_parameters": ["pre_test_power", "post_test_power"],
            "default_values": {},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": -40.0 if "Thermal" in name or "Freeze" in name else 15.0,
            "temperature_range_max": 85.0,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "Pass/Fail",
            "typical_uncertainty_percent": None,
            "equipment_requirements": ["Environmental chamber", "Power measurement system"],
            "procedure_steps": ["Perform initial measurement", f"Execute {name}", "Perform final measurement", "Compare to pass criteria"],
            "pass_criteria": {"power_degradation_max": 5.0} if "Power" in name else {"visual_defects": False},
            "display_order": 50 + i,
        })

    # =========================================================================
    # IEC 61853: ENERGY RATING (6 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC61853-1-001",
            "standard_reference": "IEC 61853-1",
            "standard_version": "2011",
            "method_clause_no": "6",
            "name": "Power Matrix Measurement",
            "description": "Power measurement at multiple irradiance and temperature combinations",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["irradiance_levels", "temperature_levels", "power_matrix"],
            "optional_parameters": [],
            "default_values": {
                "irradiance_levels": [100, 200, 400, 600, 800, 1000, 1100],
                "temperature_levels": [15, 25, 50, 75]
            },
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 100.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Variable irradiance simulator", "Temperature chamber"],
            "procedure_steps": [
                "Define irradiance-temperature matrix (min 22 points)",
                "Stabilize at each condition",
                "Measure I-V curve",
                "Extract Pmax at each point",
                "Build power matrix"
            ],
            "pass_criteria": {"matrix_completeness": 22},
            "display_order": 80,
        },
        {
            "protocol_id": "IEC61853-2-001",
            "standard_reference": "IEC 61853-2",
            "standard_version": "2016",
            "method_clause_no": "5",
            "name": "Spectral Responsivity Measurement",
            "description": "Full spectral response measurement for energy rating",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["wavelengths", "responsivity"],
            "optional_parameters": ["bias_light"],
            "default_values": {},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si", "Perovskite"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±3%",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["DSR measurement system"],
            "procedure_steps": [
                "Configure DSR system",
                "Scan 300-1200nm",
                "Apply bias light",
                "Record responsivity",
                "Normalize to Isc"
            ],
            "pass_criteria": {"wavelength_range": [300, 1200]},
            "display_order": 81,
        },
        {
            "protocol_id": "IEC61853-2-002",
            "standard_reference": "IEC 61853-2",
            "standard_version": "2016",
            "method_clause_no": "6",
            "name": "Angle of Incidence Response",
            "description": "Measurement of IAM (Incidence Angle Modifier) response",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["angles", "relative_response"],
            "optional_parameters": ["azimuth_angles"],
            "default_values": {"angles": [0, 10, 20, 30, 40, 50, 60, 70, 80]},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Collimated light source", "Rotation stage"],
            "procedure_steps": [
                "Align module at 0° AOI",
                "Measure reference Isc",
                "Rotate to each angle",
                "Measure Isc at each angle",
                "Calculate relative response",
                "Fit IAM model"
            ],
            "pass_criteria": {"angle_range": [0, 80]},
            "display_order": 82,
        },
        {
            "protocol_id": "IEC61853-3-001",
            "standard_reference": "IEC 61853-3",
            "standard_version": "2018",
            "method_clause_no": "5",
            "name": "Energy Rating Calculation",
            "description": "Calculate annual energy yield for reference climates",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["power_matrix", "climate_profile"],
            "optional_parameters": ["spectral_data", "iam_data"],
            "default_values": {"climate_profile": "Subtropical Coastal"},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±3%",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["Power matrix data", "Climate data"],
            "procedure_steps": [
                "Load power matrix",
                "Select climate profile",
                "Apply spectral corrections",
                "Apply angular corrections",
                "Integrate hourly energy",
                "Calculate CSER (kWh/kWp)"
            ],
            "pass_criteria": {},
            "display_order": 83,
        },
        {
            "protocol_id": "IEC61853-4-001",
            "standard_reference": "IEC 61853-4",
            "standard_version": "2018",
            "method_clause_no": "4",
            "name": "Reference Climate Profiles",
            "description": "Standard climate profiles for energy rating comparison",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["climate_type"],
            "optional_parameters": [],
            "default_values": {},
            "applicable_technologies": ["All"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "Reference",
            "typical_uncertainty_percent": None,
            "equipment_requirements": ["Climate data files"],
            "procedure_steps": [
                "Select reference climate",
                "Load hourly irradiance data",
                "Load temperature data",
                "Load spectral data if available",
                "Use for energy calculation"
            ],
            "pass_criteria": {},
            "display_order": 84,
        },
        {
            "protocol_id": "IEC61853-001",
            "standard_reference": "IEC 61853",
            "standard_version": "2018",
            "method_clause_no": "All",
            "name": "Complete Energy Rating",
            "description": "Full IEC 61853 energy rating procedure combining all parts",
            "category": ProtocolCategory.ENERGY_RATING,
            "test_type": "Energy Rating",
            "required_parameters": ["power_matrix", "spectral_response", "iam_data", "climate_profiles"],
            "optional_parameters": ["bifaciality_factor"],
            "default_values": {},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 75.0,
            "irradiance_range_min": 100.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±3%",
            "typical_uncertainty_percent": 3.0,
            "equipment_requirements": ["Complete test facility"],
            "procedure_steps": [
                "Measure power matrix (IEC 61853-1)",
                "Measure spectral response (IEC 61853-2)",
                "Measure IAM (IEC 61853-2)",
                "Calculate energy for each climate (IEC 61853-3)",
                "Report CSER values"
            ],
            "pass_criteria": {},
            "display_order": 85,
        },
    ])

    # =========================================================================
    # IEC TS 60904-1-2: BIFACIALITY (3 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IEC60904-1-2-001",
            "standard_reference": "IEC TS 60904-1-2",
            "standard_version": "2019",
            "method_clause_no": "5",
            "name": "Bifaciality Factor Measurement",
            "description": "Measurement of bifaciality factor (rear/front power ratio)",
            "category": ProtocolCategory.BIFACIALITY,
            "test_type": "Bifaciality",
            "required_parameters": ["pmax_front", "pmax_rear"],
            "optional_parameters": ["irradiance_front", "irradiance_rear"],
            "default_values": {"irradiance_front": 1000, "irradiance_rear": 1000},
            "applicable_technologies": ["HJT", "TOPCon", "c-Si bifacial", "PERC+"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Dual-source simulator", "Light blocker"],
            "procedure_steps": [
                "Measure front-side I-V at STC",
                "Block front illumination",
                "Measure rear-side I-V at STC",
                "Calculate bifaciality = Pmax_rear/Pmax_front",
                "Report Isc bifaciality and Pmax bifaciality"
            ],
            "pass_criteria": {"bifaciality_min": 0.65},
            "display_order": 90,
        },
        {
            "protocol_id": "IEC60904-1-2-002",
            "standard_reference": "IEC TS 60904-1-2",
            "standard_version": "2019",
            "method_clause_no": "6",
            "name": "Bifacial Gain Estimation",
            "description": "Estimation of energy gain from rear-side irradiance",
            "category": ProtocolCategory.BIFACIALITY,
            "test_type": "Bifaciality",
            "required_parameters": ["bifaciality_factor", "ground_albedo", "mounting_height"],
            "optional_parameters": ["row_spacing", "tilt_angle"],
            "default_values": {"ground_albedo": 0.25, "mounting_height": 1.0},
            "applicable_technologies": ["HJT", "TOPCon", "c-Si bifacial", "PERC+"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±5%",
            "typical_uncertainty_percent": 5.0,
            "equipment_requirements": ["Bifaciality data", "Site parameters"],
            "procedure_steps": [
                "Input bifaciality factor",
                "Define installation parameters",
                "Calculate rear irradiance from albedo",
                "Estimate bifacial gain percentage",
                "Apply to energy yield calculation"
            ],
            "pass_criteria": {},
            "display_order": 91,
        },
        {
            "protocol_id": "IEC60904-1-2-003",
            "standard_reference": "IEC TS 60904-1-2",
            "standard_version": "2019",
            "method_clause_no": "7",
            "name": "Equivalent Irradiance Method",
            "description": "Bifacial testing using equivalent front irradiance",
            "category": ProtocolCategory.BIFACIALITY,
            "test_type": "Bifaciality",
            "required_parameters": ["front_irradiance", "rear_irradiance", "bifaciality_factor"],
            "optional_parameters": [],
            "default_values": {},
            "applicable_technologies": ["HJT", "TOPCon", "c-Si bifacial", "PERC+"],
            "temperature_range_min": 20.0,
            "temperature_range_max": 30.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["Front and rear irradiance sensors"],
            "procedure_steps": [
                "Measure front irradiance G_front",
                "Measure rear irradiance G_rear",
                "Calculate G_eq = G_front + BF * G_rear",
                "Use equivalent irradiance for power correction"
            ],
            "pass_criteria": {},
            "display_order": 92,
        },
    ])

    # =========================================================================
    # IAM ANALYSIS (3 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "IAM-001",
            "standard_reference": "IEC 61853-2",
            "standard_version": "2016",
            "method_clause_no": "Annex A",
            "name": "IAM - Physical Model",
            "description": "Physical IAM model based on Fresnel reflections",
            "category": ProtocolCategory.IAM,
            "test_type": "IAM Analysis",
            "required_parameters": ["angles", "glass_refractive_index"],
            "optional_parameters": ["ar_coating"],
            "default_values": {"glass_refractive_index": 1.526, "ar_coating": False},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±1%",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Glass properties data"],
            "procedure_steps": [
                "Define glass refractive index",
                "Calculate Fresnel reflection at each angle",
                "Account for AR coating if present",
                "Generate IAM curve"
            ],
            "pass_criteria": {},
            "display_order": 95,
        },
        {
            "protocol_id": "IAM-002",
            "standard_reference": "IEC 61853-2",
            "standard_version": "2016",
            "method_clause_no": "Annex A",
            "name": "IAM - ASHRAE Model",
            "description": "ASHRAE empirical IAM model: IAM = 1 - b0*(1/cos(θ) - 1)",
            "category": ProtocolCategory.IAM,
            "test_type": "IAM Analysis",
            "required_parameters": ["angles", "b0_coefficient"],
            "optional_parameters": [],
            "default_values": {"b0_coefficient": 0.05},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±2%",
            "typical_uncertainty_percent": 2.0,
            "equipment_requirements": ["b0 coefficient from measurement or datasheet"],
            "procedure_steps": [
                "Determine b0 coefficient",
                "Calculate IAM at each angle",
                "IAM = 1 - b0*(1/cos(θ) - 1)",
                "Apply for angles 0-90°"
            ],
            "pass_criteria": {},
            "display_order": 96,
        },
        {
            "protocol_id": "IAM-003",
            "standard_reference": "IEC 61853-2",
            "standard_version": "2016",
            "method_clause_no": "Annex A",
            "name": "IAM - Martin-Ruiz Model",
            "description": "Martin-Ruiz analytical IAM model for detailed angular response",
            "category": ProtocolCategory.IAM,
            "test_type": "IAM Analysis",
            "required_parameters": ["angles", "ar_coefficient"],
            "optional_parameters": ["c1", "c2"],
            "default_values": {"ar_coefficient": 0.16, "c1": 4.0, "c2": -0.074},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS", "a-Si"],
            "temperature_range_min": None,
            "temperature_range_max": None,
            "irradiance_range_min": None,
            "irradiance_range_max": None,
            "accuracy_class": "±1%",
            "typical_uncertainty_percent": 1.0,
            "equipment_requirements": ["Model coefficients"],
            "procedure_steps": [
                "Define ar coefficient and model parameters",
                "Calculate angular losses",
                "Apply analytical model",
                "Generate IAM curve"
            ],
            "pass_criteria": {},
            "display_order": 97,
        },
    ])

    # =========================================================================
    # HOTSPOT DETECTION (3 protocols)
    # =========================================================================
    protocols.extend([
        {
            "protocol_id": "HOTSPOT-001",
            "standard_reference": "IEC 61215-2",
            "standard_version": "2021",
            "method_clause_no": "MQT 09",
            "name": "Hotspot Detection - Thermal Imaging",
            "description": "IR thermal imaging for hotspot detection under operation",
            "category": ProtocolCategory.HOTSPOT_DETECTION,
            "test_type": "Hotspot Detection",
            "required_parameters": ["thermal_image", "ambient_temperature"],
            "optional_parameters": ["irradiance", "module_current"],
            "default_values": {"irradiance": 1000},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 85.0,
            "irradiance_range_min": 800.0,
            "irradiance_range_max": 1100.0,
            "accuracy_class": "±2°C",
            "typical_uncertainty_percent": None,
            "equipment_requirements": ["IR thermal camera", "Solar simulator or outdoor"],
            "procedure_steps": [
                "Operate module at maximum power point",
                "Allow thermal stabilization",
                "Capture thermal image",
                "Identify cells exceeding threshold",
                "Calculate delta-T from average"
            ],
            "pass_criteria": {"max_delta_t": 20.0},
            "display_order": 100,
        },
        {
            "protocol_id": "HOTSPOT-002",
            "standard_reference": "IEC 61215-2",
            "standard_version": "2021",
            "method_clause_no": "MQT 09",
            "name": "Hotspot Endurance Test",
            "description": "Hot-spot endurance test with worst-case cell shading",
            "category": ProtocolCategory.HOTSPOT_DETECTION,
            "test_type": "Hotspot Detection",
            "required_parameters": ["shading_pattern", "test_duration_hours"],
            "optional_parameters": ["ambient_temperature"],
            "default_values": {"test_duration_hours": 1},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": 15.0,
            "temperature_range_max": 85.0,
            "irradiance_range_min": 1000.0,
            "irradiance_range_max": 1000.0,
            "accuracy_class": "Pass/Fail",
            "typical_uncertainty_percent": None,
            "equipment_requirements": ["Solar simulator", "Shading masks", "Thermal monitoring"],
            "procedure_steps": [
                "Identify worst-case cell for shading",
                "Apply opaque shading to cell",
                "Operate at short-circuit for 1 hour",
                "Monitor maximum temperature",
                "Verify no permanent damage",
                "Re-measure power (≤5% loss allowed)"
            ],
            "pass_criteria": {"power_degradation_max": 5.0},
            "display_order": 101,
        },
        {
            "protocol_id": "HOTSPOT-003",
            "standard_reference": "IEC TS 62446-3",
            "standard_version": "2017",
            "method_clause_no": "6",
            "name": "Field Hotspot Inspection",
            "description": "Outdoor thermal inspection for fielded PV systems",
            "category": ProtocolCategory.HOTSPOT_DETECTION,
            "test_type": "Hotspot Detection",
            "required_parameters": ["thermal_image", "irradiance"],
            "optional_parameters": ["wind_speed", "ambient_temperature"],
            "default_values": {},
            "applicable_technologies": ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"],
            "temperature_range_min": 10.0,
            "temperature_range_max": 50.0,
            "irradiance_range_min": 600.0,
            "irradiance_range_max": 1200.0,
            "accuracy_class": "±3°C",
            "typical_uncertainty_percent": None,
            "equipment_requirements": ["Handheld IR camera", "Pyranometer"],
            "procedure_steps": [
                "Verify minimum irradiance (>600 W/m²)",
                "Record ambient conditions",
                "Scan modules systematically",
                "Flag cells >10°C above average",
                "Document location and severity"
            ],
            "pass_criteria": {"hotspot_threshold": 10.0},
            "display_order": 102,
        },
    ])

    return protocols


def seed_protocols(db: Session) -> int:
    """Seed all 54 protocols into the database.

    Args:
        db: SQLAlchemy session

    Returns:
        Number of protocols seeded
    """
    protocols_data = get_all_protocols()
    seeded_count = 0
    skipped_count = 0

    print(f"\n{'='*60}")
    print("PROTOCOL SEEDING STARTED")
    print(f"{'='*60}")
    print(f"Total protocols to seed: {len(protocols_data)}")

    for protocol_dict in protocols_data:
        # Check if protocol already exists
        existing = db.query(Protocol).filter(
            Protocol.protocol_id == protocol_dict["protocol_id"]
        ).first()

        if existing:
            skipped_count += 1
            continue

        # Create new protocol
        protocol = Protocol(**protocol_dict)
        db.add(protocol)
        seeded_count += 1
        print(f"  + Seeded: {protocol_dict['protocol_id']} - {protocol_dict['name']}")

    # Commit the transaction
    db.commit()

    print(f"\n{'='*60}")
    print("PROTOCOL SEEDING COMPLETE")
    print(f"{'='*60}")
    print(f"Newly seeded: {seeded_count}")
    print(f"Already existed (skipped): {skipped_count}")
    print(f"Total in database: {seeded_count + skipped_count}")

    # Verify seeding
    total_count = db.query(Protocol).count()
    print(f"\nVERIFICATION: Protocol table now contains {total_count} records")

    # Assert we have all 54 protocols
    assert total_count >= 54, f"ERROR: Expected at least 54 protocols, found {total_count}"
    print(f"ASSERTION PASSED: {total_count} >= 54 protocols")

    # Log category breakdown
    print(f"\nProtocol breakdown by category:")
    for category in ProtocolCategory:
        cat_count = db.query(Protocol).filter(Protocol.category == category).count()
        if cat_count > 0:
            print(f"  - {category.value}: {cat_count}")

    print(f"{'='*60}\n")

    return seeded_count


def init_database() -> None:
    """Initialize database: create tables and seed protocols.

    This function is called on application startup to ensure
    the database schema exists and protocols are populated.
    """
    print("\n" + "="*60)
    print("DATABASE INITIALIZATION STARTED")
    print("="*60)

    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")

    # Seed protocols within get_db() context
    print("\nSeeding protocols...")
    with get_db() as db:
        seed_protocols(db)

    # Final verification
    with get_db() as db:
        protocol_count = db.query(Protocol).count()
        print(f"\nFINAL VERIFICATION: {protocol_count} protocols in database")

        # List all protocol IDs for deployment log verification
        protocols = db.query(Protocol.protocol_id, Protocol.name).order_by(Protocol.display_order).all()
        print(f"\nAll {len(protocols)} protocols:")
        for p_id, p_name in protocols:
            print(f"  [{p_id}] {p_name}")

    print("\n" + "="*60)
    print("DATABASE INITIALIZATION COMPLETE")
    print("="*60 + "\n")


# Auto-initialize when module is imported in production
if os.environ.get('AUTO_INIT_DB', 'false').lower() == 'true':
    init_database()
