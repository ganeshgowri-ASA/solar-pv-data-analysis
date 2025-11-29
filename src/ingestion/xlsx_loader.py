"""Loader for Excel files (.xlsx, .xls)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_loader import BaseLoader


class XlsxLoader(BaseLoader):
    """Load Excel files from solar simulators and analysis tools."""
    
    def __init__(self, file_path: str, sheet_name: Optional[str] = None, equipment_type: Optional[str] = None):
        super().__init__(file_path, equipment_type)
        self.sheet_name = sheet_name
    
    def load(self) -> Dict[str, Any]:
        """Load data from Excel file."""
        # Read Excel file
        if self.sheet_name:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        else:
            # Try first sheet
            df = pd.read_excel(self.file_path)
        
        # Find voltage and current columns
        voltage_col = self._find_column(df, ['voltage', 'v', 'volt'])
        current_col = self._find_column(df, ['current', 'i', 'amp', 'ampere'])
        
        if voltage_col and current_col:
            voltage = self.clean_numeric_array(df[voltage_col])
            current = self.clean_numeric_array(df[current_col])
        else:
            # Try to find numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                voltage = self.clean_numeric_array(df[numeric_cols[0]])
                current = self.clean_numeric_array(df[numeric_cols[1]])
            else:
                # Check if data is in rows instead of columns
                voltage, current = self._try_row_based_format(df)
        
        # Extract metadata
        self._extract_metadata_from_excel(df)
        
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
    
    def list_sheets(self) -> list:
        """List all sheet names in Excel file."""
        xl_file = pd.ExcelFile(self.file_path)
        return xl_file.sheet_names
    
    def _find_column(self, df: pd.DataFrame, keywords: list) -> str:
        """Find column by keywords."""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None
    
    def _try_row_based_format(self, df: pd.DataFrame) -> tuple:
        """Try to extract data from row-based format."""
        # Look for "Voltage" and "Current" labels in first column
        for idx, row in df.iterrows():
            first_cell = str(row.iloc[0]).lower()
            if 'voltage' in first_cell:
                voltage = self.clean_numeric_array(row.iloc[1:])
            elif 'current' in first_cell:
                current = self.clean_numeric_array(row.iloc[1:])
        
        return voltage, current
    
    def _extract_metadata_from_excel(self, df: pd.DataFrame):
        """Extract metadata from Excel file."""
        # Check first few rows for metadata
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            for col_idx, value in enumerate(row):
                if pd.notna(value):
                    value_str = str(value).lower()
                    
                    if 'irrad' in value_str:
                        try:
                            self.metadata['irradiance'] = float(row.iloc[col_idx + 1])
                        except:
                            pass
                    
                    elif 'temp' in value_str:
                        try:
                            self.metadata['temperature'] = float(row.iloc[col_idx + 1])
                        except:
                            pass
                    
                    elif 'module' in value_str or 'sample' in value_str:
                        try:
                            self.metadata['module_id'] = str(row.iloc[col_idx + 1])
                        except:
                            pass
