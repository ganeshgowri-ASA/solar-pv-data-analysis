"""PostgreSQL Database Schema for Solar PV Test Data Analysis.

LIMS-ready schema with comprehensive tracking:
- Equipment & calibration management
- Sample & project tracking
- Personnel accountability
- Standards & methods compliance
- Environmental monitoring
- Quality control with auto-deviation highlighting
- Uncertainty propagation
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, CheckConstraint, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# ============================================================================
# ENUMERATIONS
# ============================================================================

class EquipmentStatus(enum.Enum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    CALIBRATION_DUE = "calibration_due"
    OUT_OF_SERVICE = "out_of_service"
    RETIRED = "retired"

class TestStatus(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW = "review"
    APPROVED = "approved"

class DeviationSeverity(enum.Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

# ============================================================================
# EQUIPMENT & CALIBRATION
# ============================================================================

class Equipment(Base):
    __tablename__ = 'equipment'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(String(50), unique=True, nullable=False, index=True)
    qr_code = Column(String(100), unique=True, nullable=False)
    
    manufacturer = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    serial_number = Column(String(100), unique=True, nullable=False)
    
    # Calibration tracking
    calibration_date = Column(DateTime)
    calibration_interval_days = Column(Integer, default=365)
    next_calibration_due = Column(DateTime, index=True)
    calibration_certificate_no = Column(String(100))
    
    # Maintenance tracking
    maintenance_date = Column(DateTime)
    maintenance_interval_days = Column(Integer, default=180)
    next_maintenance_due = Column(DateTime)
    
    # Status
    equipment_status = Column(Enum(EquipmentStatus), default=EquipmentStatus.ACTIVE)
    
    # Technical specifications
    irradiance_range_min = Column(Float)
    irradiance_range_max = Column(Float)
    temporal_stability = Column(Float)  # ±%
    spatial_uniformity = Column(Float)  # ±%
    spectral_match_class = Column(String(10))  # A, B, C
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    calibration_records = relationship("CalibrationRecord", back_populates="equipment")
    tests = relationship("Test", back_populates="equipment")

class CalibrationRecord(Base):
    __tablename__ = 'calibration_records'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(Integer, ForeignKey('equipment.id'), nullable=False)
    
    calibration_date = Column(DateTime, nullable=False)
    calibration_lab = Column(String(200))
    certificate_no = Column(String(100), unique=True)
    
    # Calibration results
    irradiance_correction_factor = Column(Float)
    temperature_correction_factor = Column(Float)
    uncertainty_k2 = Column(Float)  # Expanded uncertainty (k=2)
    
    # Reference standards
    reference_cell_id = Column(String(100))
    reference_traceability = Column(Text)
    
    next_calibration_due = Column(DateTime)
    performed_by = Column(String(100))
    
    calibration_data = Column(JSON)  # Detailed calibration data
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    equipment = relationship("Equipment", back_populates="calibration_records")

# ============================================================================
# PROJECTS & SAMPLES
# ============================================================================

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(String(50), unique=True, nullable=False, index=True)
    
    project_name = Column(String(200), nullable=False)
    customer_name = Column(String(200))
    customer_contact = Column(String(200))
    
    project_start_datetime = Column(DateTime, nullable=False)
    project_end_datetime = Column(DateTime)
    
    project_status = Column(Enum(TestStatus), default=TestStatus.PENDING)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    samples = relationship("Sample", back_populates="project")

class Sample(Base):
    __tablename__ = 'samples'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), unique=True, nullable=False, index=True)
    qr_code = Column(String(100), unique=True, nullable=False)
    
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    
    # Sample details
    manufacturer = Column(String(100))
    model = Column(String(100))
    serial_number = Column(String(100))
    technology = Column(String(50))  # c-Si, HJT, TOPCon, CdTe, CIGS, etc.
    
    # Module specifications
    rated_power = Column(Float)  # Wp
    rated_voltage = Column(Float)  # V
    rated_current = Column(Float)  # A
    cell_count = Column(Integer)
    cell_configuration = Column(String(50))  # 60-cell, 72-cell, half-cut, etc.
    
    # Physical dimensions
    length_mm = Column(Float)
    width_mm = Column(Float)
    thickness_mm = Column(Float)
    weight_kg = Column(Float)
    
    # Receiving information
    received_date = Column(DateTime)
    condition_on_receipt = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    project = relationship("Project", back_populates="samples")
    tests = relationship("Test", back_populates="sample")

# ============================================================================
# PERSONNEL
# ============================================================================

class Personnel(Base):
    __tablename__ = 'personnel'
    
    id = Column(Integer, primary_key=True)
    technician_id = Column(String(50), unique=True, nullable=False, index=True)
    
    technician_name = Column(String(100), nullable=False)
    email = Column(String(100))
    phone = Column(String(50))
    
    role = Column(String(50))  # Technician, Reviewer, Manager, etc.
    certifications = Column(JSON)  # List of certifications
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tests_performed = relationship("Test", foreign_keys="Test.technician_id", back_populates="technician")
    tests_reviewed = relationship("Test", foreign_keys="Test.reviewer_id", back_populates="reviewer")

# ============================================================================
# TESTS & MEASUREMENTS
# ============================================================================

class Test(Base):
    __tablename__ = 'tests'
    
    id = Column(Integer, primary_key=True)
    test_id = Column(String(50), unique=True, nullable=False, index=True)
    
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    equipment_id = Column(Integer, ForeignKey('equipment.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    
    # Test type
    test_type = Column(String(100), nullable=False)  # I-V curve, temperature coefficient, etc.
    test_datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Standards & methods
    standard_reference = Column(String(100))  # IEC 60904-1, IEC 61215, etc.
    method_clause_no = Column(String(50))
    correction_procedure = Column(String(50))  # Procedure 1, 2, 3, 4
    
    # Reference material
    reference_material_used = Column(String(100))
    reference_cell_id = Column(String(100))
    
    # Environmental conditions
    ambient_temperature = Column(Float)  # °C
    relative_humidity = Column(Float)  # %
    atmospheric_pressure = Column(Float)  # hPa
    irradiance_level = Column(Float)  # W/m²
    module_temperature = Column(Float)  # °C
    
    # Personnel
    technician_id = Column(Integer, ForeignKey('personnel.id'))
    reviewer_id = Column(Integer, ForeignKey('personnel.id'))
    
    # Status
    test_status = Column(Enum(TestStatus), default=TestStatus.PENDING)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    sample = relationship("Sample", back_populates="tests")
    equipment = relationship("Equipment", back_populates="tests")
    technician = relationship("Personnel", foreign_keys=[technician_id], back_populates="tests_performed")
    reviewer = relationship("Personnel", foreign_keys=[reviewer_id], back_populates="tests_reviewed")
    
    iv_measurements = relationship("IVMeasurement", back_populates="test")
    quality_control = relationship("QualityControl", back_populates="test", uselist=False)

# ============================================================================
# I-V MEASUREMENT DATA
# ============================================================================

class IVMeasurement(Base):
    __tablename__ = 'iv_measurements'
    
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey('tests.id'), nullable=False)
    
    # Measurement conditions
    measured_irradiance = Column(Float, nullable=False)  # W/m²
    measured_temperature = Column(Float, nullable=False)  # °C
    
    # Raw I-V data
    voltage_array = Column(JSON, nullable=False)  # Array of voltage points
    current_array = Column(JSON, nullable=False)  # Array of current points
    
    # Extracted parameters (raw)
    isc_raw = Column(Float)  # Short-circuit current (A)
    voc_raw = Column(Float)  # Open-circuit voltage (V)
    pmax_raw = Column(Float)  # Maximum power (W)
    vmpp_raw = Column(Float)  # Voltage at MPP (V)
    impp_raw = Column(Float)  # Current at MPP (A)
    fill_factor_raw = Column(Float)  # Fill factor
    
    # STC-corrected parameters
    isc_stc = Column(Float)
    voc_stc = Column(Float)
    pmax_stc = Column(Float)
    vmpp_stc = Column(Float)
    impp_stc = Column(Float)
    fill_factor_stc = Column(Float)
    
    # Temperature coefficients
    alpha_isc = Column(Float)  # A/°C
    beta_voc = Column(Float)  # V/°C
    gamma_pmax = Column(Float)  # W/°C
    
    # Series resistance
    series_resistance = Column(Float)  # Ohms
    
    # Curve correction factor (spectral mismatch)
    kappa = Column(Float)
    
    # Spectral mismatch factor
    mismatch_factor = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("Test", back_populates="iv_measurements")

# ============================================================================
# QUALITY CONTROL
# ============================================================================

class QualityControl(Base):
    __tablename__ = 'quality_control'
    
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey('tests.id'), nullable=False, unique=True)
    
    # Control chart data
    control_chart_data = Column(JSON)  # Historical data for control charts
    
    # Pass/Fail criteria
    pass_criteria = Column(JSON, nullable=False)  # Dict of parameter: threshold
    test_result = Column(String(20))  # PASS, FAIL, CONDITIONAL
    
    # Tolerances
    tolerances = Column(JSON)  # Dict of parameter: tolerance
    
    # Uncertainties (GUM)
    uncertainties = Column(JSON)  # Dict of parameter: uncertainty
    combined_uncertainty = Column(Float)  # Combined standard uncertainty
    expanded_uncertainty = Column(Float)  # Expanded uncertainty (k=2)
    coverage_factor = Column(Float, default=2.0)  # k factor
    
    # Statement of conformity
    statement_of_conformity = Column(Text)
    conformity_status = Column(Boolean)  # True if conforms to specification
    
    # Auto deviation highlighting
    deviations_detected = Column(JSON)  # List of detected deviations
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    test = relationship("Test", back_populates="quality_control")
    deviation_flags = relationship("DeviationFlag", back_populates="quality_control")

class DeviationFlag(Base):
    __tablename__ = 'deviation_flags'
    
    id = Column(Integer, primary_key=True)
    qc_id = Column(Integer, ForeignKey('quality_control.id'), nullable=False)
    
    parameter_name = Column(String(100), nullable=False)
    
    # Values
    measured_value = Column(Float, nullable=False)
    prescribed_value = Column(Float, nullable=False)
    tolerance = Column(Float)
    
    # Deviation calculation
    deviation_absolute = Column(Float)  # Absolute deviation
    deviation_percentage = Column(Float)  # Percentage deviation
    
    # Flag status
    deviation_flag = Column(Boolean, default=False)  # Auto-calculated
    deviation_severity = Column(Enum(DeviationSeverity), default=DeviationSeverity.NORMAL)
    
    # Thresholds
    warning_threshold = Column(Float)  # % threshold for warning
    critical_threshold = Column(Float)  # % threshold for critical
    
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    quality_control = relationship("QualityControl", back_populates="deviation_flags")
    
    __table_args__ = (
        Index('idx_deviation_severity', 'deviation_severity'),
        CheckConstraint('deviation_percentage >= 0', name='check_positive_deviation'),
    )

# ============================================================================
# INDEXES FOR PERFORMANCE
# ============================================================================

Index('idx_test_datetime', Test.test_datetime)
Index('idx_test_status', Test.test_status)
Index('idx_sample_technology', Sample.technology)
Index('idx_equipment_status', Equipment.equipment_status)
Index('idx_calibration_due', Equipment.next_calibration_due)
