"""Database module for PostgreSQL integration.

LIMS features:
- Equipment tracking with QR codes
- Sample management
- Personnel accountability
- Auto-deviation detection
- Uncertainty propagation
- Protocol management with IEC standards
"""

from .schema import (
    Base,
    Equipment,
    Sample,
    Test,
    IVMeasurement,
    QualityControl,
    Protocol,
    ProtocolCategory,
    CalibrationRecord,
    Project,
    Personnel,
    DeviationFlag,
    EquipmentStatus,
    TestStatus,
    DeviationSeverity,
)
from .init_db import (
    init_database,
    get_db,
    seed_protocols,
    get_all_protocols,
    engine,
    SessionLocal,
    DATABASE_URL,
)

__all__ = [
    # Base
    'Base',
    # Models
    'Equipment',
    'Sample',
    'Test',
    'IVMeasurement',
    'QualityControl',
    'Protocol',
    'CalibrationRecord',
    'Project',
    'Personnel',
    'DeviationFlag',
    # Enums
    'ProtocolCategory',
    'EquipmentStatus',
    'TestStatus',
    'DeviationSeverity',
    # Database functions
    'init_database',
    'get_db',
    'seed_protocols',
    'get_all_protocols',
    'engine',
    'SessionLocal',
    'DATABASE_URL',
]
