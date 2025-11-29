"""Database module for PostgreSQL integration.

LIMS features:
- Equipment tracking with QR codes
- Sample management
- Personnel accountability
- Auto-deviation detection
- Uncertainty propagation
"""

from .schema import Base, Equipment, Sample, Test, IVMeasurement, QualityControl
from .init_db import init_database

__all__ = ['Base', 'Equipment', 'Sample', 'Test', 'IVMeasurement', 'QualityControl', 'init_database']
