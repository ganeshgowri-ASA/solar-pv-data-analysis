"""I-V Curve Analysis Module (IEC 60904-1).

Extracts key performance parameters from I-V curve data.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IVCurveAnalyzer:
    """Analyze I-V curves and extract performance parameters."""
    
    def __init__(self, voltage: np.ndarray, current: np.ndarray):
        """Initialize with I-V data.
        
        Args:
            voltage: Array of voltage values (V)
            current: Array of current values (A)
        """
        self.voltage = np.array(voltage)
        self.current = np.array(current)
        
        # Sort by voltage
        sort_idx = np.argsort(self.voltage)
        self.voltage = self.voltage[sort_idx]
        self.current = self.current[sort_idx]
        
        # Results storage
        self.results = {}
    
    def extract_parameters(self) -> Dict:
        """Extract all I-V curve parameters.
        
        Returns:
            Dictionary containing:
                - isc: Short-circuit current (A)
                - voc: Open-circuit voltage (V)
                - pmax: Maximum power (W)
                - vmpp: Voltage at MPP (V)
                - impp: Current at MPP (A)
                - ff: Fill factor
                - efficiency: Module efficiency (if area provided)
        """
        # Extract basic parameters
        self.results['isc'] = self._find_isc()
        self.results['voc'] = self._find_voc()
        
        # Find maximum power point
        vmpp, impp, pmax = self._find_mpp()
        self.results['vmpp'] = vmpp
        self.results['impp'] = impp
        self.results['pmax'] = pmax
        
        # Calculate fill factor
        self.results['ff'] = self._calculate_fill_factor()
        
        # Calculate series and shunt resistance
        self.results['rs'] = self._estimate_series_resistance()
        self.results['rsh'] = self._estimate_shunt_resistance()
        
        return self.results
    
    def _find_isc(self) -> float:
        """Find short-circuit current (current at V=0)."""
        # Find closest point to V=0
        idx_zero = np.argmin(np.abs(self.voltage))
        
        if np.abs(self.voltage[idx_zero]) < 0.1:  # Within 100mV
            return self.current[idx_zero]
        else:
            # Interpolate to find Isc at V=0
            f = interp1d(self.voltage, self.current, kind='linear', fill_value='extrapolate')
            return float(f(0.0))
    
    def _find_voc(self) -> float:
        """Find open-circuit voltage (voltage at I=0)."""
        # Find closest point to I=0
        idx_zero = np.argmin(np.abs(self.current))
        
        if np.abs(self.current[idx_zero]) < 0.01:  # Within 10mA
            return self.voltage[idx_zero]
        else:
            # Interpolate to find Voc at I=0
            f = interp1d(self.current[::-1], self.voltage[::-1], kind='linear', fill_value='extrapolate')
            return float(f(0.0))
    
    def _find_mpp(self) -> Tuple[float, float, float]:
        """Find maximum power point.
        
        Returns:
            (vmpp, impp, pmax)
        """
        # Calculate power at each point
        power = self.voltage * self.current
        
        # Find maximum
        idx_max = np.argmax(power)
        
        vmpp = self.voltage[idx_max]
        impp = self.current[idx_max]
        pmax = power[idx_max]
        
        # Refine using interpolation
        if idx_max > 0 and idx_max < len(power) - 1:
            # Use parabolic interpolation for better accuracy
            v_range = self.voltage[idx_max-1:idx_max+2]
            p_range = power[idx_max-1:idx_max+2]
            
            # Fit parabola
            try:
                coeffs = np.polyfit(v_range, p_range, 2)
                # Find vertex of parabola
                vmpp_refined = -coeffs[1] / (2 * coeffs[0])
                
                # Ensure refined value is reasonable
                if v_range[0] <= vmpp_refined <= v_range[2]:
                    vmpp = vmpp_refined
                    # Interpolate current
                    f_current = interp1d(self.voltage, self.current, kind='quadratic')
                    impp = float(f_current(vmpp))
                    pmax = vmpp * impp
            except:
                pass  # Use original values if refinement fails
        
        return vmpp, impp, pmax
    
    def _calculate_fill_factor(self) -> float:
        """Calculate fill factor.
        
        FF = (Vmpp * Impp) / (Voc * Isc)
        """
        isc = self.results.get('isc')
        voc = self.results.get('voc')
        pmax = self.results.get('pmax')
        
        if isc and voc and pmax:
            ff = pmax / (voc * isc)
            return min(ff, 1.0)  # FF should not exceed 1.0
        
        return 0.0
    
    def _estimate_series_resistance(self) -> float:
        """Estimate series resistance (Rs) from I-V curve.
        
        Uses slope near Voc (dV/dI at I=0).
        """
        # Use points near Voc (last 10% of curve)
        n_points = max(5, len(self.voltage) // 10)
        v_near_voc = self.voltage[-n_points:]
        i_near_voc = self.current[-n_points:]
        
        # Calculate slope dV/dI
        if len(v_near_voc) > 2:
            slope = np.polyfit(i_near_voc, v_near_voc, 1)[0]
            rs = abs(slope)
            return rs if rs < 10.0 else 0.5  # Typical range check
        
        return 0.5  # Default value
    
    def _estimate_shunt_resistance(self) -> float:
        """Estimate shunt resistance (Rsh) from I-V curve.
        
        Uses slope near Isc (dI/dV at V=0).
        """
        # Use points near Isc (first 10% of curve)
        n_points = max(5, len(self.voltage) // 10)
        v_near_isc = self.voltage[:n_points]
        i_near_isc = self.current[:n_points]
        
        # Calculate slope dI/dV
        if len(v_near_isc) > 2:
            slope = np.polyfit(v_near_isc, i_near_isc, 1)[0]
            rsh = abs(1.0 / slope) if slope != 0 else 1000.0
            return rsh if rsh < 10000.0 else 1000.0  # Typical range check
        
        return 1000.0  # Default value
    
    def interpolate_curve(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth I-V curve by interpolation.
        
        Args:
            num_points: Number of points in interpolated curve
        
        Returns:
            (voltage_interp, current_interp)
        """
        v_min, v_max = self.voltage.min(), self.voltage.max()
        v_interp = np.linspace(v_min, v_max, num_points)
        
        f = interp1d(self.voltage, self.current, kind='cubic', fill_value='extrapolate')
        i_interp = f(v_interp)
        
        # Ensure current stays positive
        i_interp = np.maximum(i_interp, 0)
        
        return v_interp, i_interp
    
    def get_power_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power curve (P-V).
        
        Returns:
            (voltage, power)
        """
        power = self.voltage * self.current
        return self.voltage, power
    
    def summary(self) -> str:
        """Generate summary report."""
        if not self.results:
            self.extract_parameters()
        
        summary = f"""
I-V Curve Analysis Results
========================

Short-Circuit Current (Isc): {self.results['isc']:.3f} A
Open-Circuit Voltage (Voc):  {self.results['voc']:.3f} V

Maximum Power Point:
  Voltage (Vmpp):  {self.results['vmpp']:.3f} V
  Current (Impp):  {self.results['impp']:.3f} A
  Power (Pmax):    {self.results['pmax']:.3f} W

Fill Factor (FF):  {self.results['ff']:.4f}

Series Resistance (Rs):   {self.results['rs']:.4f} Ω
Shunt Resistance (Rsh):   {self.results['rsh']:.1f} Ω
"""
        return summary
