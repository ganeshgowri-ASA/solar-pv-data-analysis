"""Bifaciality Analysis Module per IEC TS 60904-1-2.

Implements bifaciality factor calculation, rear-side contribution analysis,
and bifacial gain estimation for bifacial PV modules.

IEC TS 60904-1-2: Photovoltaic devices - Part 1-2: Measurement of current-voltage
characteristics of bifacial photovoltaic (PV) devices

Key Parameters:
- Bifaciality Factor (φ): Ratio of rear to front side efficiency
- Bifacial Gain (BG): Additional energy from rear-side irradiance
- Ground Coverage Ratio (GCR): Module area / ground area
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class MeasurementSide(Enum):
    """Measurement side for bifacial testing."""
    FRONT = "front"
    REAR = "rear"
    BIFACIAL = "bifacial"


class MountingConfiguration(Enum):
    """Typical bifacial mounting configurations."""
    GROUND_MOUNT_FIXED = "ground_mount_fixed"
    GROUND_MOUNT_TRACKER = "ground_mount_tracker"
    ELEVATED_FIXED = "elevated_fixed"
    ELEVATED_TRACKER = "elevated_tracker"
    VERTICAL_EW = "vertical_east_west"
    CARPORT = "carport"
    AGRIVOLTAIC = "agrivoltaic"


@dataclass
class BifacialIVData:
    """I-V data for bifacial measurement."""
    voltage: np.ndarray         # V
    current: np.ndarray         # A
    irradiance_front: float     # W/m² on front
    irradiance_rear: float      # W/m² on rear (0 for front-only)
    temperature: float          # °C
    side: MeasurementSide

    # Extracted parameters (filled after analysis)
    isc: Optional[float] = None
    voc: Optional[float] = None
    pmax: Optional[float] = None
    ff: Optional[float] = None
    vmpp: Optional[float] = None
    impp: Optional[float] = None


@dataclass
class BifacialityResult:
    """Results of bifaciality factor measurement."""
    # Bifaciality factors
    phi_isc: float      # Isc bifaciality factor (φ_Isc)
    phi_voc: float      # Voc bifaciality factor (φ_Voc)
    phi_pmax: float     # Pmax bifaciality factor (φ_Pmax)
    phi_ff: float       # FF bifaciality factor (φ_FF)

    # Front and rear parameters
    isc_front: float
    isc_rear: float
    voc_front: float
    voc_rear: float
    pmax_front: float
    pmax_rear: float
    ff_front: float
    ff_rear: float

    # Test conditions
    irradiance_front: float
    irradiance_rear: float
    temperature: float


@dataclass
class BifacialGainResult:
    """Results of bifacial gain calculation."""
    bifacial_gain: float        # % gain from rear side
    rear_contribution: float    # % of total power from rear
    effective_irradiance: float # W/m² total effective
    power_monofacial: float     # W (front only)
    power_bifacial: float       # W (front + rear contribution)
    ground_reflected: float     # W/m² reflected from ground
    rear_irradiance_ratio: float  # Rear/Front irradiance ratio


class BifacialAnalyzer:
    """Analyzer for bifacial PV module measurements per IEC TS 60904-1-2."""

    # Typical bifaciality factors by technology
    TYPICAL_BIFACIALITY = {
        'mono_perc': {'phi_pmax': 0.70, 'phi_isc': 0.70, 'range': (0.65, 0.75)},
        'mono_topcon': {'phi_pmax': 0.85, 'phi_isc': 0.85, 'range': (0.80, 0.90)},
        'mono_hjt': {'phi_pmax': 0.92, 'phi_isc': 0.92, 'range': (0.88, 0.95)},
        'multi_perc': {'phi_pmax': 0.65, 'phi_isc': 0.65, 'range': (0.60, 0.70)},
        'n_type_pert': {'phi_pmax': 0.80, 'phi_isc': 0.80, 'range': (0.75, 0.85)},
    }

    # Typical ground albedo values
    GROUND_ALBEDO = {
        'grass': 0.20,
        'concrete': 0.30,
        'sand': 0.35,
        'snow': 0.80,
        'white_gravel': 0.50,
        'dark_soil': 0.10,
        'water': 0.08,
        'asphalt': 0.12,
    }

    def __init__(
        self,
        module_area: float,
        pmax_stc_front: float,
        n_cells: int = 144,
        technology: str = 'mono_perc'
    ):
        """Initialize bifacial analyzer.

        Args:
            module_area: Module area in m²
            pmax_stc_front: Front-side STC power in W
            n_cells: Number of cells
            technology: Cell technology
        """
        self.module_area = module_area
        self.pmax_stc_front = pmax_stc_front
        self.n_cells = n_cells
        self.technology = technology

        self._front_data = None
        self._rear_data = None
        self._bifaciality_result = None

    def add_measurement(self, data: BifacialIVData):
        """Add I-V measurement data.

        Args:
            data: Bifacial I-V data
        """
        # Extract parameters
        data = self._extract_parameters(data)

        if data.side == MeasurementSide.FRONT:
            self._front_data = data
        elif data.side == MeasurementSide.REAR:
            self._rear_data = data

    def _extract_parameters(self, data: BifacialIVData) -> BifacialIVData:
        """Extract I-V parameters from measurement."""
        v = data.voltage
        i = data.current

        # Find Isc (current at V=0)
        idx_0v = np.argmin(np.abs(v))
        data.isc = i[idx_0v]

        # Find Voc (voltage at I=0)
        idx_0i = np.argmin(np.abs(i))
        data.voc = v[idx_0i]

        # Find MPP
        power = v * i
        idx_mpp = np.argmax(power)
        data.pmax = power[idx_mpp]
        data.vmpp = v[idx_mpp]
        data.impp = i[idx_mpp]

        # Calculate FF
        if data.isc > 0 and data.voc > 0:
            data.ff = data.pmax / (data.isc * data.voc)
        else:
            data.ff = 0

        return data

    def calculate_bifaciality(
        self,
        front_data: Optional[BifacialIVData] = None,
        rear_data: Optional[BifacialIVData] = None
    ) -> BifacialityResult:
        """Calculate bifaciality factors per IEC TS 60904-1-2.

        Bifaciality factor φ = (Rear side value) / (Front side value)
        at equivalent irradiance and temperature conditions.

        Args:
            front_data: Front-side measurement data
            rear_data: Rear-side measurement data

        Returns:
            BifacialityResult with all bifaciality factors
        """
        front = front_data or self._front_data
        rear = rear_data or self._rear_data

        if front is None or rear is None:
            raise ValueError("Both front and rear measurements required")

        # Ensure parameters are extracted
        if front.pmax is None:
            front = self._extract_parameters(front)
        if rear.pmax is None:
            rear = self._extract_parameters(rear)

        # Normalize to same irradiance (1000 W/m²)
        irr_factor_front = 1000 / front.irradiance_front if front.irradiance_front > 0 else 1
        irr_factor_rear = 1000 / rear.irradiance_rear if rear.irradiance_rear > 0 else 1

        # Corrected values at 1000 W/m²
        isc_front_norm = front.isc * irr_factor_front
        isc_rear_norm = rear.isc * irr_factor_rear

        # Voc correction (logarithmic with irradiance)
        voc_front_norm = front.voc + 0.026 * self.n_cells * np.log(irr_factor_front)
        voc_rear_norm = rear.voc + 0.026 * self.n_cells * np.log(irr_factor_rear)

        # Power and FF
        pmax_front_norm = front.pmax * irr_factor_front
        pmax_rear_norm = rear.pmax * irr_factor_rear

        # Calculate bifaciality factors
        phi_isc = isc_rear_norm / isc_front_norm if isc_front_norm > 0 else 0
        phi_voc = voc_rear_norm / voc_front_norm if voc_front_norm > 0 else 0
        phi_pmax = pmax_rear_norm / pmax_front_norm if pmax_front_norm > 0 else 0
        phi_ff = rear.ff / front.ff if front.ff > 0 else 0

        self._bifaciality_result = BifacialityResult(
            phi_isc=phi_isc,
            phi_voc=phi_voc,
            phi_pmax=phi_pmax,
            phi_ff=phi_ff,
            isc_front=front.isc,
            isc_rear=rear.isc,
            voc_front=front.voc,
            voc_rear=rear.voc,
            pmax_front=front.pmax,
            pmax_rear=rear.pmax,
            ff_front=front.ff,
            ff_rear=rear.ff,
            irradiance_front=front.irradiance_front,
            irradiance_rear=rear.irradiance_rear,
            temperature=(front.temperature + rear.temperature) / 2
        )

        return self._bifaciality_result

    def calculate_bifacial_gain(
        self,
        front_irradiance: float,
        rear_irradiance: float,
        phi_pmax: Optional[float] = None,
        temperature: float = 25.0,
        temp_coeff: float = -0.35
    ) -> BifacialGainResult:
        """Calculate bifacial gain for given irradiance conditions.

        Bifacial Gain = (P_bifacial - P_monofacial) / P_monofacial × 100%

        Args:
            front_irradiance: Front-side irradiance (W/m²)
            rear_irradiance: Rear-side irradiance (W/m²)
            phi_pmax: Bifaciality factor (uses measured or typical if None)
            temperature: Module temperature (°C)
            temp_coeff: Temperature coefficient of Pmax (%/°C)

        Returns:
            BifacialGainResult
        """
        # Get bifaciality factor
        if phi_pmax is None:
            if self._bifaciality_result is not None:
                phi_pmax = self._bifaciality_result.phi_pmax
            else:
                typical = self.TYPICAL_BIFACIALITY.get(
                    self.technology,
                    self.TYPICAL_BIFACIALITY['mono_perc']
                )
                phi_pmax = typical['phi_pmax']

        # Temperature correction factor
        temp_factor = 1 + (temp_coeff / 100) * (temperature - 25)

        # Front-side power
        power_front = self.pmax_stc_front * (front_irradiance / 1000) * temp_factor

        # Rear-side contribution
        rear_contribution_power = self.pmax_stc_front * phi_pmax * (rear_irradiance / 1000) * temp_factor

        # Total bifacial power
        power_bifacial = power_front + rear_contribution_power

        # Bifacial gain
        bifacial_gain = ((power_bifacial - power_front) / power_front * 100) if power_front > 0 else 0

        # Rear contribution percentage
        rear_pct = (rear_contribution_power / power_bifacial * 100) if power_bifacial > 0 else 0

        # Effective irradiance
        effective_irr = front_irradiance + phi_pmax * rear_irradiance

        return BifacialGainResult(
            bifacial_gain=bifacial_gain,
            rear_contribution=rear_pct,
            effective_irradiance=effective_irr,
            power_monofacial=power_front,
            power_bifacial=power_bifacial,
            ground_reflected=rear_irradiance,
            rear_irradiance_ratio=rear_irradiance / front_irradiance if front_irradiance > 0 else 0
        )

    def estimate_rear_irradiance(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        albedo: float = 0.20,
        height: float = 1.0,
        gcr: float = 0.4,
        tilt: float = 20.0,
        solar_elevation: float = 45.0
    ) -> Dict[str, float]:
        """Estimate rear-side irradiance using view factor model.

        Args:
            ghi: Global horizontal irradiance (W/m²)
            dni: Direct normal irradiance (W/m²)
            dhi: Diffuse horizontal irradiance (W/m²)
            albedo: Ground albedo (reflectivity)
            height: Module height above ground (m)
            gcr: Ground coverage ratio
            tilt: Module tilt angle (degrees)
            solar_elevation: Solar elevation angle (degrees)

        Returns:
            Dictionary with irradiance components
        """
        # Ground-reflected irradiance
        ground_reflected = ghi * albedo

        # View factor from module rear to ground
        # Simplified model based on height and tilt
        tilt_rad = np.radians(tilt)

        # View factor approximation
        # Higher modules and lower tilt see more ground
        height_factor = min(1.0, height / 2.0)
        tilt_factor = np.cos(tilt_rad)
        row_factor = 1 - gcr * 0.5  # Row shading effect

        vf_ground = 0.5 * (1 + np.cos(tilt_rad)) * height_factor * row_factor

        # Rear irradiance from ground reflection
        rear_from_ground = ground_reflected * vf_ground

        # Diffuse sky contribution to rear (small for tilted modules)
        rear_from_sky = dhi * 0.5 * (1 - np.cos(tilt_rad)) * 0.3

        # Direct component (edge effects, generally small)
        solar_elev_rad = np.radians(solar_elevation)
        if solar_elevation < tilt:
            # Sun can hit rear during low sun angles
            rear_direct = dni * 0.1 * np.sin(tilt_rad - solar_elev_rad)
        else:
            rear_direct = 0

        total_rear = rear_from_ground + rear_from_sky + rear_direct

        # Front irradiance estimate
        front_poa = dni * np.cos(np.radians(tilt - solar_elevation)) + dhi * (1 + np.cos(tilt_rad)) / 2
        front_poa = max(front_poa, 0)

        return {
            'rear_total': total_rear,
            'rear_from_ground': rear_from_ground,
            'rear_from_sky': rear_from_sky,
            'rear_direct': rear_direct,
            'front_poa': front_poa,
            'rear_front_ratio': total_rear / front_poa if front_poa > 0 else 0,
            'ground_reflected': ground_reflected,
            'view_factor_ground': vf_ground
        }

    def annual_bifacial_gain(
        self,
        phi_pmax: Optional[float] = None,
        albedo: float = 0.20,
        gcr: float = 0.4,
        height: float = 1.0,
        mounting: MountingConfiguration = MountingConfiguration.GROUND_MOUNT_FIXED
    ) -> Dict[str, float]:
        """Estimate annual bifacial gain for installation configuration.

        Args:
            phi_pmax: Bifaciality factor
            albedo: Ground albedo
            gcr: Ground coverage ratio
            height: Module height above ground (m)
            mounting: Mounting configuration

        Returns:
            Dictionary with annual gain estimates
        """
        if phi_pmax is None:
            if self._bifaciality_result is not None:
                phi_pmax = self._bifaciality_result.phi_pmax
            else:
                typical = self.TYPICAL_BIFACIALITY.get(self.technology, {})
                phi_pmax = typical.get('phi_pmax', 0.70)

        # Base rear irradiance ratio estimates by mounting type
        base_ratios = {
            MountingConfiguration.GROUND_MOUNT_FIXED: 0.08,
            MountingConfiguration.GROUND_MOUNT_TRACKER: 0.12,
            MountingConfiguration.ELEVATED_FIXED: 0.15,
            MountingConfiguration.ELEVATED_TRACKER: 0.18,
            MountingConfiguration.VERTICAL_EW: 0.25,
            MountingConfiguration.CARPORT: 0.20,
            MountingConfiguration.AGRIVOLTAIC: 0.15,
        }

        base_ratio = base_ratios.get(mounting, 0.10)

        # Adjust for albedo (normalized to 0.20)
        albedo_factor = albedo / 0.20

        # Adjust for height
        height_factor = min(1.5, 0.5 + height / 2.0)

        # Adjust for GCR (lower GCR = less row shading = more rear irradiance)
        gcr_factor = 1 + (0.4 - gcr) * 0.5

        # Final rear irradiance ratio
        rear_ratio = base_ratio * albedo_factor * height_factor * gcr_factor

        # Annual bifacial gain
        annual_gain = phi_pmax * rear_ratio * 100  # %

        # Energy-weighted gain (typically lower due to mismatch)
        energy_gain = annual_gain * 0.85  # Typical mismatch factor

        return {
            'annual_bifacial_gain_percent': annual_gain,
            'energy_weighted_gain_percent': energy_gain,
            'average_rear_irradiance_ratio': rear_ratio,
            'bifaciality_factor': phi_pmax,
            'albedo': albedo,
            'gcr': gcr,
            'height_m': height,
            'mounting_type': mounting.value,
            'optimal_height_m': 1.5 if mounting in [
                MountingConfiguration.GROUND_MOUNT_FIXED,
                MountingConfiguration.GROUND_MOUNT_TRACKER
            ] else 3.0
        }

    def get_summary_report(self) -> str:
        """Generate summary report of bifaciality analysis."""
        if self._bifaciality_result is None:
            return "No bifaciality analysis performed yet."

        r = self._bifaciality_result

        typical = self.TYPICAL_BIFACIALITY.get(self.technology, {})
        expected_range = typical.get('range', (0.65, 0.85))

        status = "PASS" if expected_range[0] <= r.phi_pmax <= expected_range[1] else "REVIEW"

        report = f"""
Bifaciality Analysis Report (IEC TS 60904-1-2)
===============================================

Module Technology: {self.technology}
Module Area: {self.module_area:.2f} m²
Front Side Pmax (STC): {self.pmax_stc_front:.1f} W

Test Conditions:
---------------
Front Irradiance: {r.irradiance_front:.0f} W/m²
Rear Irradiance: {r.irradiance_rear:.0f} W/m²
Temperature: {r.temperature:.1f}°C

Measured Parameters:
-------------------
                Front Side    Rear Side
Isc (A):        {r.isc_front:>10.3f}    {r.isc_rear:>10.3f}
Voc (V):        {r.voc_front:>10.2f}    {r.voc_rear:>10.2f}
Pmax (W):       {r.pmax_front:>10.1f}    {r.pmax_rear:>10.1f}
FF:             {r.ff_front:>10.4f}    {r.ff_rear:>10.4f}

Bifaciality Factors:
-------------------
φ_Isc:  {r.phi_isc:.3f}  (Isc bifaciality)
φ_Voc:  {r.phi_voc:.3f}  (Voc bifaciality)
φ_Pmax: {r.phi_pmax:.3f}  (Pmax bifaciality) ← Primary metric
φ_FF:   {r.phi_ff:.3f}  (FF bifaciality)

Assessment:
----------
Expected range for {self.technology}: {expected_range[0]:.2f} - {expected_range[1]:.2f}
Status: {status}
"""

        return report


def calculate_bifaciality_factor(
    pmax_front: float,
    pmax_rear: float,
    irr_front: float = 1000,
    irr_rear: float = 1000
) -> float:
    """Quick calculation of bifaciality factor.

    Args:
        pmax_front: Front-side power (W)
        pmax_rear: Rear-side power (W)
        irr_front: Front-side irradiance (W/m²)
        irr_rear: Rear-side irradiance (W/m²)

    Returns:
        Bifaciality factor (φ)
    """
    # Normalize to same irradiance
    pmax_front_norm = pmax_front * (1000 / irr_front) if irr_front > 0 else pmax_front
    pmax_rear_norm = pmax_rear * (1000 / irr_rear) if irr_rear > 0 else pmax_rear

    return pmax_rear_norm / pmax_front_norm if pmax_front_norm > 0 else 0


def estimate_bifacial_gain(
    phi: float,
    rear_irradiance_ratio: float = 0.10
) -> float:
    """Estimate bifacial gain from bifaciality factor.

    Args:
        phi: Bifaciality factor
        rear_irradiance_ratio: Rear/Front irradiance ratio

    Returns:
        Bifacial gain as percentage
    """
    return phi * rear_irradiance_ratio * 100


def get_typical_bifaciality(technology: str) -> Dict[str, float]:
    """Get typical bifaciality values for a technology.

    Args:
        technology: Cell technology name

    Returns:
        Dictionary with typical bifaciality parameters
    """
    return BifacialAnalyzer.TYPICAL_BIFACIALITY.get(
        technology,
        BifacialAnalyzer.TYPICAL_BIFACIALITY['mono_perc']
    )


def get_ground_albedo(surface: str) -> float:
    """Get typical ground albedo value.

    Args:
        surface: Surface type

    Returns:
        Albedo value
    """
    return BifacialAnalyzer.GROUND_ALBEDO.get(surface, 0.20)
