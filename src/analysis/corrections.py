"""IEC 60891 Correction Procedures.

Translate I-V curves to Standard Test Conditions (STC):
- Temperature: 25°C
- Irradiance: 1000 W/m²
- Air Mass: AM1.5G
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class STCConditions:
    """Standard Test Conditions."""
    temperature: float = 25.0  # °C
    irradiance: float = 1000.0  # W/m²
    air_mass: float = 1.5


class CorrectionProcedure1:
    """IEC 60891 Procedure 1 - Full correction with all coefficients.
    
    Most accurate method requiring:
    - Alpha: Temperature coefficient of Isc (A/°C)
    - Beta: Temperature coefficient of Voc (V/°C)
    - Rs: Series resistance (Ω)
    - Kappa: Curve correction factor (Ω/°C)
    """
    
    def __init__(self, alpha: float, beta: float, rs: float, kappa: float):
        """Initialize with temperature coefficients.
        
        Args:
            alpha: Temperature coefficient of Isc (A/°C)
            beta: Temperature coefficient of Voc (V/°C)
            rs: Series resistance (Ω)
            kappa: Curve correction factor (Ω/°C)
        """
        self.alpha = alpha
        self.beta = beta
        self.rs = rs
        self.kappa = kappa
        self.stc = STCConditions()
    
    def correct_to_stc(self, 
                       voltage: np.ndarray,
                       current: np.ndarray,
                       isc_measured: float,
                       temperature: float,
                       irradiance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Correct I-V curve to STC.
        
        Args:
            voltage: Measured voltage array (V)
            current: Measured current array (A)
            isc_measured: Measured short-circuit current (A)
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        
        Returns:
            (voltage_stc, current_stc): Corrected I-V curve at STC
        """
        # Temperature difference
        delta_t = self.stc.temperature - temperature
        
        # Irradiance ratio
        g_ratio = self.stc.irradiance / irradiance
        
        # Current correction (Equation from IEC 60891)
        # I_2 = I_1 + Isc_1 * (G_2/G_1 - 1) + alpha * delta_T
        current_stc = current + isc_measured * (g_ratio - 1) + self.alpha * delta_t
        
        # Voltage correction (Equation from IEC 60891)
        # V_2 = V_1 - Rs * (I_2 - I_1) - kappa * I_2 * delta_T + beta * delta_T
        voltage_stc = (voltage - 
                      self.rs * (current_stc - current) -
                      self.kappa * current_stc * delta_t +
                      self.beta * delta_t)
        
        return voltage_stc, current_stc
    
    def correct_parameters(self,
                          isc: float,
                          voc: float,
                          pmax: float,
                          temperature: float,
                          irradiance: float) -> Dict[str, float]:
        """Correct individual parameters to STC.
        
        Args:
            isc: Measured short-circuit current (A)
            voc: Measured open-circuit voltage (V)
            pmax: Measured maximum power (W)
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        
        Returns:
            Dictionary with corrected parameters
        """
        delta_t = self.stc.temperature - temperature
        g_ratio = self.stc.irradiance / irradiance
        
        # Correct Isc
        isc_stc = isc * g_ratio + self.alpha * delta_t
        
        # Correct Voc
        voc_stc = voc + self.beta * delta_t
        
        # Approximate Pmax correction
        # More accurate: use corrected I-V curve
        pmax_stc = pmax * g_ratio * (1 + (self.alpha/isc + self.beta/voc) * delta_t)
        
        return {
            'isc_stc': isc_stc,
            'voc_stc': voc_stc,
            'pmax_stc': pmax_stc
        }


class CorrectionProcedure2:
    """IEC 60891 Procedure 2 - Correction using multiple curves.
    
    Uses interpolation between measurements at different conditions.
    Requires at least 3 measurements at different irradiances/temperatures.
    """
    
    def __init__(self):
        self.stc = STCConditions()
        self.measurements = []
    
    def add_measurement(self, 
                       voltage: np.ndarray,
                       current: np.ndarray,
                       temperature: float,
                       irradiance: float):
        """Add a measurement to the dataset.
        
        Args:
            voltage: Measured voltage array
            current: Measured current array
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        """
        self.measurements.append({
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'irradiance': irradiance
        })
    
    def correct_to_stc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Correct to STC using interpolation.
        
        Returns:
            (voltage_stc, current_stc): Interpolated curve at STC
        """
        if len(self.measurements) < 3:
            raise ValueError("Procedure 2 requires at least 3 measurements")
        
        # Implementation would involve multi-dimensional interpolation
        # Simplified version: weight by distance to STC
        weights = []
        for m in self.measurements:
            dist = np.sqrt((m['temperature'] - self.stc.temperature)**2 +
                          ((m['irradiance'] - self.stc.irradiance)/10)**2)
            weights.append(1.0 / (dist + 1e-6))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of curves
        voltage_stc = sum(w * m['voltage'] for w, m in zip(weights, self.measurements))
        current_stc = sum(w * m['current'] for w, m in zip(weights, self.measurements))
        
        return voltage_stc, current_stc


class CorrectionProcedure3:
    """IEC 60891 Procedure 3 - Simplified correction with Rs.
    
    Simplified version of Procedure 1, assuming kappa = 0.
    Suitable for crystalline silicon modules.
    """
    
    def __init__(self, alpha: float, beta: float, rs: float):
        """Initialize with simplified coefficients.
        
        Args:
            alpha: Temperature coefficient of Isc (A/°C)
            beta: Temperature coefficient of Voc (V/°C)
            rs: Series resistance (Ω)
        """
        self.alpha = alpha
        self.beta = beta
        self.rs = rs
        self.stc = STCConditions()
    
    def correct_to_stc(self,
                       voltage: np.ndarray,
                       current: np.ndarray,
                       isc_measured: float,
                       temperature: float,
                       irradiance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Correct I-V curve to STC (simplified).
        
        Args:
            voltage: Measured voltage array (V)
            current: Measured current array (A)
            isc_measured: Measured short-circuit current (A)
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        
        Returns:
            (voltage_stc, current_stc): Corrected I-V curve at STC
        """
        delta_t = self.stc.temperature - temperature
        g_ratio = self.stc.irradiance / irradiance
        
        # Current correction (same as Procedure 1)
        current_stc = current + isc_measured * (g_ratio - 1) + self.alpha * delta_t
        
        # Voltage correction (simplified - no kappa term)
        voltage_stc = voltage - self.rs * (current_stc - current) + self.beta * delta_t
        
        return voltage_stc, current_stc


class CorrectionProcedure4:
    """IEC 60891 Procedure 4 - Quick correction without Rs.
    
    Simplest method for field measurements.
    Assumes Rs = 0 and kappa = 0.
    """
    
    def __init__(self, alpha: float, beta: float):
        """Initialize with basic temperature coefficients.
        
        Args:
            alpha: Temperature coefficient of Isc (A/°C)
            beta: Temperature coefficient of Voc (V/°C)
        """
        self.alpha = alpha
        self.beta = beta
        self.stc = STCConditions()
    
    def correct_to_stc(self,
                       voltage: np.ndarray,
                       current: np.ndarray,
                       isc_measured: float,
                       temperature: float,
                       irradiance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Correct I-V curve to STC (quick method).
        
        Args:
            voltage: Measured voltage array (V)
            current: Measured current array (A)
            isc_measured: Measured short-circuit current (A)
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        
        Returns:
            (voltage_stc, current_stc): Corrected I-V curve at STC
        """
        delta_t = self.stc.temperature - temperature
        g_ratio = self.stc.irradiance / irradiance
        
        # Current correction
        current_stc = current + isc_measured * (g_ratio - 1) + self.alpha * delta_t
        
        # Voltage correction (no Rs correction)
        voltage_stc = voltage + self.beta * delta_t
        
        return voltage_stc, current_stc
    
    def correct_parameters(self,
                          isc: float,
                          voc: float,
                          temperature: float,
                          irradiance: float) -> Dict[str, float]:
        """Quick correction of parameters.
        
        Args:
            isc: Measured short-circuit current (A)
            voc: Measured open-circuit voltage (V)
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)
        
        Returns:
            Dictionary with corrected parameters
        """
        delta_t = self.stc.temperature - temperature
        g_ratio = self.stc.irradiance / irradiance
        
        isc_stc = isc * g_ratio + self.alpha * delta_t
        voc_stc = voc + self.beta * delta_t
        
        return {
            'isc_stc': isc_stc,
            'voc_stc': voc_stc
        }
