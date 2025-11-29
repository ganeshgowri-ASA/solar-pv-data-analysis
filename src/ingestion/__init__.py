"""Data Ingestion Module for Solar PV Test Data.

Supports multiple equipment manufacturers and file formats:

Equipment:
- PASAN, Spire, Halm, MBJ, Wavelabs, Endeas, Avalon, Quicksun, G-Solar

File Formats:
- CSV (comma/semicolon separated)
- TXT (tab-delimited)
- XLSX/XLS (Excel)
- Spectral response files

Features:
- Auto-detection of equipment from filename/content
- Universal file parsers with equipment-specific configurations
- Metadata extraction from headers
- Data validation and cleaning
"""

from .base_loader import BaseLoader
from .auto_detector import AutoDetector
from .txt_loader import TxtLoader
from .csv_loader import CsvLoader
from .xlsx_loader import XlsxLoader

__all__ = [
    "BaseLoader",
    "AutoDetector",
    "TxtLoader",
    "CsvLoader",
    "XlsxLoader",
]
