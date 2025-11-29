"""Base loader class for all data file types."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np


class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, file_path: str, equipment_type: Optional[str] = None):
        self.file_path = Path(file_path)
        self.equipment_type = equipment_type
        self.data = None
        self.metadata = {}
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load data from file.
        
        Returns:
            Dict containing:
                - 'voltage': array of voltage values
                - 'current': array of current values
                - 'metadata': dict of additional information
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate loaded data."""
        pass
    
    def get_file_extension(self) -> str:
        """Get file extension."""
        return self.file_path.suffix.lower()
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from file."""
        return self.metadata
    
    @staticmethod
    def clean_numeric_array(arr: List) -> np.ndarray:
        """Clean and convert to numeric array."""
        # Remove NaN and invalid values
        arr = pd.to_numeric(arr, errors='coerce')
        arr = arr[~np.isnan(arr)]
        return np.array(arr)
    
    @staticmethod
    def detect_delimiter(file_path: Path) -> str:
        """Detect delimiter in text file."""
        with open(file_path, 'r') as f:
            first_line = f.readline()
            
        # Check common delimiters
        delimiters = [',', '\t', ';', '|']
        delimiter_counts = {d: first_line.count(d) for d in delimiters}
        
        # Return delimiter with highest count
        return max(delimiter_counts, key=delimiter_counts.get)
