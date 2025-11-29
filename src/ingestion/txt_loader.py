"""Loader for tab-delimited text files (.txt)."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_loader import BaseLoader


class TxtLoader(BaseLoader):
    """Load tab-delimited text files from solar simulators."""
    
    def load(self) -> Dict[str, Any]:
        """Load data from TXT file.
        
        Supports formats like:
        - Spire: tab-delimited with headers
        - Generic: Voltage\tCurrent format
        """
        # Detect delimiter
        delimiter = self.detect_delimiter(self.file_path)
        
        # Try loading with pandas
        try:
            # Read with header
            df = pd.read_csv(self.file_path, delimiter=delimiter)
            
            # Find voltage and current columns
            voltage_col = self._find_column(df, ['voltage', 'v', 'volt'])
            current_col = self._find_column(df, ['current', 'i', 'amp', 'ampere'])
            
            if voltage_col and current_col:
                voltage = self.clean_numeric_array(df[voltage_col])
                current = self.clean_numeric_array(df[current_col])
            else:
                # Try first two columns
                voltage = self.clean_numeric_array(df.iloc[:, 0])
                current = self.clean_numeric_array(df.iloc[:, 1])
            
            # Extract metadata from headers if available
            self._extract_metadata_from_df(df)
            
        except Exception as e:
            # Fallback: load as simple two-column format
            data = np.loadtxt(self.file_path, delimiter=delimiter)
            voltage = data[:, 0]
            current = data[:, 1]
        
        self.data = {
            'voltage': voltage,
            'current': current,
            'metadata': self.metadata
        }
        
        return self.data
    
    def validate(self) -> bool:
        """Validate loaded data."""
        if self.data is None:
            return False
        
        voltage = self.data.get('voltage')
        current = self.data.get('current')
        
        if voltage is None or current is None:
            return False
        
        if len(voltage) != len(current):
            return False
        
        if len(voltage) < 10:  # Minimum points
            return False
        
        # Check voltage range (0 to ~100V typical)
        if np.min(voltage) < -1 or np.max(voltage) > 150:
            return False
        
        # Check current range
        if np.min(current) < -1 or np.max(current) > 50:
            return False
        
        return True
    
    def _find_column(self, df: pd.DataFrame, keywords: list) -> str:
        """Find column by keywords."""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None
    
    def _extract_metadata_from_df(self, df: pd.DataFrame):
        """Extract metadata from dataframe columns."""
        # Look for metadata in column names or first rows
        for col in df.columns:
            col_str = str(col).lower()
            if 'irrad' in col_str and 'irradiance' not in self.metadata:
                try:
                    self.metadata['irradiance'] = float(df[col].iloc[0])
                except:
                    pass
            elif 'temp' in col_str and 'temperature' not in self.metadata:
                try:
                    self.metadata['temperature'] = float(df[col].iloc[0])
                except:
                    pass
