"""Configuration file for Solar PV Data Analysis Platform.

Paths, environment variables, and database connections.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COMPUTED_DATA_DIR = DATA_DIR / "computed"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, COMPUTED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_CONFIG: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "pv_test_data"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Railway PostgreSQL connection string format
RAILWAY_DB_URL = os.getenv("DATABASE_URL", "")

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================
APP_CONFIG = {
    "app_name": "Solar PV Test Analysis Platform",
    "version": "1.0.0",
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}

# ============================================================================
# FILE UPLOAD SETTINGS
# ============================================================================
UPLOAD_CONFIG = {
    "max_file_size_mb": 100,
    "allowed_extensions": [".txt", ".csv", ".xlsx", ".xls"],
    "chunk_size": 8192,
}

# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================
ANALYSIS_CONFIG = {
    "default_stc_temperature": 25.0,  # °C
    "default_stc_irradiance": 1000.0,  # W/m²
    "default_am": 1.5,  # Air Mass
    "interpolation_points": 1000,
    "uncertainty_confidence_level": 0.95,
}

# ============================================================================
# REPORTING SETTINGS
# ============================================================================
REPORT_CONFIG = {
    "company_name": "PV Testing Laboratory",
    "logo_path": None,
    "accreditation": "ISO/IEC 17025:2017",
    "export_formats": ["pdf", "docx", "xlsx"],
}
