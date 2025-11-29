"""Loader for CSV files (.csv)."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_loader import BaseLoader


class CsvLoader(BaseLoader):
    """Load CSV files from solar simulators."""
    
    def load(self) -> Dict[str, Any]:
        """Load data from CSV file."""
        # Read CSV with pandas
        df = pd.read_csv(self.file_path)
        
        # Find voltage and current columns
        voltage_col = self._find_column(df, ['voltage', 'v', 'volt'])
        current_col = self._find_column(df, ['current', 'i', 'amp', 'ampere'])
        
        if voltage_col and current_col:
            voltage = self.clean_numeric_array(df[voltage_col])
            current = self.clean_numeric_array(df[current_col])
        else:
            # Try first two numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                voltage = self.clean_numeric_array(df[numeric_cols[0]])
                current = self.clean_numeric_array(df[numeric_cols[1]])
            else:
                raise ValueError("Could not find voltage and current columns")
        
        # Extract metadata
        self._extract_metadata_from_df(df)
        
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
        
        if len(voltage) < 10:
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
        """Extract metadata from dataframe."""
        # Check for metadata columns
        metadata_keywords = {
            'irradiance': ['irrad', 'irradiance', 'g'],
            'temperature': ['temp', 'temperature', 't'],
            'module_id': ['module', 'sample', 'id'],
        }
        
        for meta_key, keywords in metadata_keywords.items():
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in keywords):
                    try:
                        value = df[col].iloc[0]
                        if pd.notna(value):
                            self.metadata[meta_key] = value
                    except:
                        pass
