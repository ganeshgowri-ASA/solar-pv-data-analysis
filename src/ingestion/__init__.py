"""Data ingestion module for various flasher/simulator formats.

Supports multiple equipment manufacturers:
- PASAN, Spire, Halm, MBJ, G-Solar, Endeas, Avalon

File formats:
- CSV, XLSX, TXT (tab-delimited)
- Spectral response files
"""

from .base_loader import BaseLoader
from .auto_detector import AutoDetector

__all__ = ['BaseLoader', 'AutoDetector']
