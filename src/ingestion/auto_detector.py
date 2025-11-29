"""Auto-detect equipment type and file format."""

import re
from pathlib import Path
from typing import Optional, Dict, Any
from .txt_loader import TxtLoader
from .csv_loader import CsvLoader
from .xlsx_loader import XlsxLoader


class AutoDetector:
    """Auto-detect equipment type and load data accordingly."""
    
    EQUIPMENT_PATTERNS = {
        'PASAN': [r'pasan', r'highlight', r'hl-\d+'],
        'SPIRE': [r'spire', r'spi-sun', r'sp-\d+'],
        'HALM': [r'halm', r'flashsim', r'he-\d+'],
        'MBJ': [r'meyer.*burger', r'mbj', r'wavelabs', r'sinus'],
        'GSOLAR': [r'g-?solar', r'quicksun', r'gs-\d+'],
        'ENDEAS': [r'endeas', r'en-\d+'],
        'AVALON': [r'avalon', r'av-\d+'],
    }
    
    @classmethod
    def detect_equipment(cls, file_path: str) -> Optional[str]:
        """Detect equipment type from filename or file content."""
        file_path = Path(file_path)
        filename = file_path.name.lower()
        
        # Check filename patterns
        for equipment, patterns in cls.EQUIPMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename):
                    return equipment
        
        # Check file content (first few lines)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = ''.join([f.readline().lower() for _ in range(10)])
            
            for equipment, patterns in cls.EQUIPMENT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, content):
                        return equipment
        except:
            pass
        
        return None
    
    @classmethod
    def load_file(cls, file_path: str, equipment_type: Optional[str] = None) -> Dict[str, Any]:
        """Auto-detect format and load file.
        
        Args:
            file_path: Path to data file
            equipment_type: Optional equipment type (auto-detected if not provided)
        
        Returns:
            Dictionary with voltage, current, and metadata
        """
        file_path = Path(file_path)
        
        # Detect equipment if not provided
        if equipment_type is None:
            equipment_type = cls.detect_equipment(str(file_path))
        
        # Select appropriate loader based on file extension
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            loader = TxtLoader(str(file_path), equipment_type)
        elif ext == '.csv':
            loader = CsvLoader(str(file_path), equipment_type)
        elif ext in ['.xlsx', '.xls']:
            loader = XlsxLoader(str(file_path), equipment_type=equipment_type)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Load and validate data
        data = loader.load()
        
        if not loader.validate():
            raise ValueError(f"Data validation failed for {file_path}")
        
        # Add equipment type to metadata
        data['metadata']['equipment_type'] = equipment_type
        data['metadata']['filename'] = file_path.name
        
        return data
    
    @classmethod
    def batch_load(cls, file_paths: list, equipment_type: Optional[str] = None) -> list:
        """Load multiple files.
        
        Args:
            file_paths: List of file paths
            equipment_type: Optional equipment type for all files
        
        Returns:
            List of data dictionaries
        """
        results = []
        errors = []
        
        for file_path in file_paths:
            try:
                data = cls.load_file(file_path, equipment_type)
                results.append(data)
            except Exception as e:
                errors.append({'file': file_path, 'error': str(e)})
        
        return results, errors
