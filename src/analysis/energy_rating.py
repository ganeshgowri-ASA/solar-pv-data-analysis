"""Energy Rating Module per IEC 61853.

Implements power matrix measurements, climate-specific energy rating (CSER),
climate-specific power rating (CSPR), and annual energy prediction.

IEC 61853 Series:
- IEC 61853-1: Irradiance and temperature performance measurements
- IEC 61853-2: Spectral responsivity, incidence angle, module temperature
- IEC 61853-3: Energy rating of PV modules
- IEC 61853-4: Standard reference climatic profiles
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import simpson
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class ClimateProfile(Enum):
    """IEC 61853-4 standard reference climatic profiles."""
    SUBTROPICAL_ARID = "subtropical_arid"           # Hot desert
    SUBTROPICAL_COASTAL = "subtropical_coastal"      # Hot humid
    TEMPERATE_COASTAL = "temperate_coastal"         # Marine west coast
    TEMPERATE_CONTINENTAL = "temperate_continental"  # Humid continental
    HIGH_ELEVATION = "high_elevation"               # Highland
    TROPICAL_HUMID = "tropical_humid"               # Tropical rainforest


@dataclass
class ClimateData:
    """Climate profile data for energy rating."""
    name: str
    description: str
    annual_irradiation: float  # kWh/m²/year (GHI)
    avg_ambient_temp: float    # °C
    temp_range: Tuple[float, float]  # min, max annual °C
    avg_wind_speed: float      # m/s
    spectral_factor: float     # Average spectral correction
    irradiance_distribution: Dict[int, float]  # W/m² bins -> hours/year


# Standard climate profiles per IEC 61853-4
CLIMATE_PROFILES = {
    ClimateProfile.SUBTROPICAL_ARID: ClimateData(
        name="Subtropical Arid",
        description="Hot desert climate (Phoenix, Alice Springs)",
        annual_irradiation=2400,
        avg_ambient_temp=23.0,
        temp_range=(5, 45),
        avg_wind_speed=3.5,
        spectral_factor=1.01,
        irradiance_distribution={
            100: 800, 200: 700, 300: 600, 400: 500,
            500: 450, 600: 400, 700: 350, 800: 300,
            900: 250, 1000: 200, 1100: 100
        }
    ),
    ClimateProfile.SUBTROPICAL_COASTAL: ClimateData(
        name="Subtropical Coastal",
        description="Hot humid climate (Miami, Hong Kong)",
        annual_irradiation=1800,
        avg_ambient_temp=25.0,
        temp_range=(15, 35),
        avg_wind_speed=4.0,
        spectral_factor=0.99,
        irradiance_distribution={
            100: 900, 200: 750, 300: 600, 400: 500,
            500: 400, 600: 350, 700: 300, 800: 250,
            900: 150, 1000: 100, 1100: 50
        }
    ),
    ClimateProfile.TEMPERATE_COASTAL: ClimateData(
        name="Temperate Coastal",
        description="Marine west coast (London, Seattle)",
        annual_irradiation=1100,
        avg_ambient_temp=11.0,
        temp_range=(-5, 25),
        avg_wind_speed=4.5,
        spectral_factor=0.98,
        irradiance_distribution={
            100: 1000, 200: 800, 300: 600, 400: 450,
            500: 350, 600: 250, 700: 200, 800: 150,
            900: 100, 1000: 50, 1100: 20
        }
    ),
    ClimateProfile.TEMPERATE_CONTINENTAL: ClimateData(
        name="Temperate Continental",
        description="Humid continental (Berlin, Chicago)",
        annual_irradiation=1200,
        avg_ambient_temp=10.0,
        temp_range=(-15, 35),
        avg_wind_speed=3.8,
        spectral_factor=0.99,
        irradiance_distribution={
            100: 950, 200: 750, 300: 550, 400: 450,
            500: 400, 600: 300, 700: 250, 800: 200,
            900: 120, 1000: 70, 1100: 30
        }
    ),
    ClimateProfile.HIGH_ELEVATION: ClimateData(
        name="High Elevation",
        description="Highland climate (Denver, La Paz)",
        annual_irradiation=2000,
        avg_ambient_temp=8.0,
        temp_range=(-10, 30),
        avg_wind_speed=4.2,
        spectral_factor=1.02,
        irradiance_distribution={
            100: 700, 200: 600, 300: 550, 400: 500,
            500: 450, 600: 400, 700: 350, 800: 300,
            900: 250, 1000: 200, 1100: 150
        }
    ),
    ClimateProfile.TROPICAL_HUMID: ClimateData(
        name="Tropical Humid",
        description="Tropical rainforest (Singapore, Jakarta)",
        annual_irradiation=1700,
        avg_ambient_temp=27.0,
        temp_range=(23, 33),
        avg_wind_speed=2.5,
        spectral_factor=0.98,
        irradiance_distribution={
            100: 850, 200: 700, 300: 600, 400: 500,
            500: 450, 600: 400, 700: 350, 800: 280,
            900: 180, 1000: 90, 1100: 30
        }
    )
}


@dataclass
class PowerMatrixPoint:
    """Single measurement point in the power matrix."""
    irradiance: float   # W/m²
    temperature: float  # °C
    pmax: float         # W
    voc: float          # V
    isc: float          # A
    vmpp: float         # V
    impp: float         # A
    ff: float           # Fill factor


@dataclass
class PowerMatrix:
    """IEC 61853-1 Power Matrix."""
    # Standard irradiance levels (W/m²)
    irradiance_levels: List[float] = field(default_factory=lambda: [
        100, 200, 400, 600, 800, 1000, 1100
    ])

    # Standard temperature levels (°C)
    temperature_levels: List[float] = field(default_factory=lambda: [
        15, 25, 50, 75
    ])

    # Power matrix data: Dict[(irradiance, temperature)] -> PowerMatrixPoint
    data: Dict[Tuple[float, float], PowerMatrixPoint] = field(default_factory=dict)

    # Module specifications
    pmax_stc: float = 0.0
    area: float = 0.0  # m²

    def add_point(self, point: PowerMatrixPoint):
        """Add a measurement point to the matrix."""
        key = (point.irradiance, point.temperature)
        self.data[key] = point

    def get_power(self, irradiance: float, temperature: float) -> Optional[float]:
        """Get power at specific conditions (interpolated if needed)."""
        key = (irradiance, temperature)
        if key in self.data:
            return self.data[key].pmax

        # Interpolate
        return self._interpolate_power(irradiance, temperature)

    def _interpolate_power(self, irradiance: float, temperature: float) -> Optional[float]:
        """Interpolate power from matrix data."""
        if len(self.data) < 4:
            return None

        # Build interpolation grid
        irr_vals = sorted(set(k[0] for k in self.data.keys()))
        temp_vals = sorted(set(k[1] for k in self.data.keys()))

        power_grid = np.zeros((len(irr_vals), len(temp_vals)))
        for i, irr in enumerate(irr_vals):
            for j, temp in enumerate(temp_vals):
                key = (irr, temp)
                if key in self.data:
                    power_grid[i, j] = self.data[key].pmax
                else:
                    power_grid[i, j] = np.nan

        # Fill NaN values with nearest neighbor
        from scipy.ndimage import generic_filter
        mask = np.isnan(power_grid)
        power_grid[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            power_grid[~mask]
        )

        try:
            interp = RegularGridInterpolator(
                (irr_vals, temp_vals),
                power_grid,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            result = interp([[irradiance, temperature]])[0]
            return float(result)
        except Exception:
            return None

    def get_efficiency_matrix(self) -> Dict[Tuple[float, float], float]:
        """Calculate efficiency at each matrix point."""
        efficiencies = {}
        for key, point in self.data.items():
            irr, temp = key
            if self.area > 0 and irr > 0:
                eff = point.pmax / (irr * self.area) * 100
                efficiencies[key] = eff
        return efficiencies

    def is_complete(self) -> bool:
        """Check if matrix has all required points."""
        required = len(self.irradiance_levels) * len(self.temperature_levels)
        return len(self.data) >= required


class EnergyRatingResult(NamedTuple):
    """Energy rating calculation results."""
    cser: float                 # Climate-Specific Energy Rating (kWh)
    cspr: float                 # Climate-Specific Power Rating (W)
    annual_energy: float        # Annual energy yield (kWh/kWp)
    performance_ratio: float    # PR
    temperature_loss: float     # % loss due to temperature
    low_irradiance_loss: float  # % loss at low irradiance
    climate_profile: str
    reference_yield: float      # kWh/kWp at STC


class EnergyRatingCalculator:
    """IEC 61853-3 Energy Rating Calculator."""

    def __init__(
        self,
        power_matrix: PowerMatrix,
        module_area: float,
        pmax_stc: float,
        temp_coeff_pmax: float = -0.35,  # %/°C
    ):
        """Initialize calculator.

        Args:
            power_matrix: IEC 61853-1 power matrix
            module_area: Module area in m²
            pmax_stc: STC power rating in W
            temp_coeff_pmax: Temperature coefficient of Pmax (%/°C)
        """
        self.power_matrix = power_matrix
        self.power_matrix.area = module_area
        self.power_matrix.pmax_stc = pmax_stc
        self.module_area = module_area
        self.pmax_stc = pmax_stc
        self.temp_coeff = temp_coeff_pmax / 100  # Convert to fraction

        self._build_interpolator()

    def _build_interpolator(self):
        """Build power interpolation from matrix."""
        if not self.power_matrix.data:
            self._interpolator = None
            return

        irr_vals = sorted(set(k[0] for k in self.power_matrix.data.keys()))
        temp_vals = sorted(set(k[1] for k in self.power_matrix.data.keys()))

        power_grid = np.zeros((len(irr_vals), len(temp_vals)))
        for i, irr in enumerate(irr_vals):
            for j, temp in enumerate(temp_vals):
                key = (irr, temp)
                if key in self.power_matrix.data:
                    power_grid[i, j] = self.power_matrix.data[key].pmax
                else:
                    # Estimate using temperature coefficient
                    stc_point = self.power_matrix.data.get((1000, 25))
                    if stc_point:
                        p_stc = stc_point.pmax
                        p_est = p_stc * (irr / 1000) * (1 + self.temp_coeff * (temp - 25))
                        power_grid[i, j] = max(0, p_est)
                    else:
                        power_grid[i, j] = self.pmax_stc * (irr / 1000)

        self._irr_vals = np.array(irr_vals)
        self._temp_vals = np.array(temp_vals)
        self._power_grid = power_grid

        try:
            self._interpolator = RegularGridInterpolator(
                (irr_vals, temp_vals),
                power_grid,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
        except Exception:
            self._interpolator = None

    def get_power(self, irradiance: float, module_temp: float) -> float:
        """Get module power at given conditions.

        Args:
            irradiance: Irradiance in W/m²
            module_temp: Module temperature in °C

        Returns:
            Power in W
        """
        if self._interpolator is not None:
            try:
                # Clip to valid range
                irr_clipped = np.clip(
                    irradiance,
                    self._irr_vals.min(),
                    self._irr_vals.max()
                )
                temp_clipped = np.clip(
                    module_temp,
                    self._temp_vals.min(),
                    self._temp_vals.max()
                )
                result = self._interpolator([[irr_clipped, temp_clipped]])[0]
                return max(0, float(result))
            except Exception:
                pass

        # Fallback: simple temperature correction from STC
        p_at_stc = self.pmax_stc * (irradiance / 1000)
        temp_correction = 1 + self.temp_coeff * (module_temp - 25)
        return max(0, p_at_stc * temp_correction)

    def calculate_module_temperature(
        self,
        irradiance: float,
        ambient_temp: float,
        wind_speed: float = 1.0,
        mounting: str = "open_rack"
    ) -> float:
        """Calculate module temperature from environmental conditions.

        Uses NOCT-based model or Faiman model.

        Args:
            irradiance: Irradiance in W/m²
            ambient_temp: Ambient temperature in °C
            wind_speed: Wind speed in m/s
            mounting: Mounting type ("open_rack", "roof_mount", "bipv")

        Returns:
            Module temperature in °C
        """
        # Faiman model coefficients
        coefficients = {
            "open_rack": (25.0, 8.0),      # U0, U1
            "roof_mount": (22.0, 6.0),
            "bipv": (20.0, 4.0)
        }

        u0, u1 = coefficients.get(mounting, (25.0, 8.0))

        # Faiman equation
        # T_module = T_ambient + G / (U0 + U1 * v)
        u_total = u0 + u1 * wind_speed
        delta_t = irradiance / u_total

        return ambient_temp + delta_t

    def calculate_cser(
        self,
        climate: ClimateProfile,
        spectral_factor: float = 1.0,
        iam_factor: float = 0.97,
        soiling_factor: float = 0.98
    ) -> EnergyRatingResult:
        """Calculate Climate-Specific Energy Rating per IEC 61853-3.

        Args:
            climate: Reference climate profile
            spectral_factor: Spectral correction factor
            iam_factor: Incidence angle modifier factor
            soiling_factor: Soiling loss factor

        Returns:
            EnergyRatingResult with CSER and related metrics
        """
        climate_data = CLIMATE_PROFILES[climate]

        # Calculate energy for each irradiance bin
        total_energy = 0.0
        total_energy_stc = 0.0
        temp_loss_energy = 0.0
        low_irr_loss_energy = 0.0

        for irr_bin, hours in climate_data.irradiance_distribution.items():
            # Estimate ambient temperature for this irradiance level
            # Higher irradiance -> higher temperature
            irr_fraction = irr_bin / 1000
            temp_range = climate_data.temp_range[1] - climate_data.temp_range[0]
            ambient_temp = (
                climate_data.avg_ambient_temp +
                (irr_fraction - 0.5) * temp_range * 0.3
            )

            # Calculate module temperature
            module_temp = self.calculate_module_temperature(
                irr_bin,
                ambient_temp,
                climate_data.avg_wind_speed
            )

            # Get power at these conditions
            power = self.get_power(irr_bin, module_temp)

            # Apply correction factors
            power_corrected = (
                power *
                spectral_factor *
                climate_data.spectral_factor *
                iam_factor *
                soiling_factor
            )

            # Calculate energy contribution (kWh)
            energy = power_corrected * hours / 1000
            total_energy += energy

            # Reference energy at STC
            energy_stc = self.pmax_stc * (irr_bin / 1000) * hours / 1000
            total_energy_stc += energy_stc

            # Temperature loss
            power_at_25c = self.get_power(irr_bin, 25.0)
            temp_loss_energy += (power_at_25c - power) * hours / 1000

            # Low irradiance loss (relative efficiency drop)
            if irr_bin < 400:
                eff_at_irr = power / (irr_bin * self.module_area) if self.module_area > 0 else 0
                eff_stc = self.pmax_stc / (1000 * self.module_area) if self.module_area > 0 else 0
                if eff_stc > 0:
                    low_irr_loss = (1 - eff_at_irr / eff_stc) * energy_stc
                    low_irr_loss_energy += max(0, low_irr_loss)

        # CSER in kWh
        cser = total_energy

        # CSPR (Climate-Specific Power Rating)
        # Average power weighted by irradiance distribution
        total_hours = sum(climate_data.irradiance_distribution.values())
        cspr = cser / total_hours * 1000 if total_hours > 0 else 0

        # Annual energy yield (kWh/kWp)
        annual_yield = cser / (self.pmax_stc / 1000) if self.pmax_stc > 0 else 0

        # Performance ratio
        pr = cser / total_energy_stc if total_energy_stc > 0 else 0

        # Loss percentages
        temp_loss_pct = temp_loss_energy / total_energy_stc * 100 if total_energy_stc > 0 else 0
        low_irr_loss_pct = low_irr_loss_energy / total_energy_stc * 100 if total_energy_stc > 0 else 0

        return EnergyRatingResult(
            cser=cser,
            cspr=cspr,
            annual_energy=annual_yield,
            performance_ratio=pr,
            temperature_loss=temp_loss_pct,
            low_irradiance_loss=low_irr_loss_pct,
            climate_profile=climate_data.name,
            reference_yield=climate_data.annual_irradiation
        )

    def calculate_annual_energy(
        self,
        irradiance_profile: np.ndarray,
        temperature_profile: np.ndarray,
        wind_speed: float = 2.0,
        time_step_hours: float = 1.0
    ) -> Dict[str, float]:
        """Calculate annual energy from hourly profiles.

        Args:
            irradiance_profile: Hourly irradiance values (W/m²)
            temperature_profile: Hourly ambient temperature values (°C)
            wind_speed: Average wind speed (m/s)
            time_step_hours: Time step in hours

        Returns:
            Dictionary with energy metrics
        """
        total_energy = 0.0
        operating_hours = 0
        peak_power = 0.0

        for irr, temp in zip(irradiance_profile, temperature_profile):
            if irr > 10:  # Threshold for operation
                module_temp = self.calculate_module_temperature(irr, temp, wind_speed)
                power = self.get_power(irr, module_temp)
                energy = power * time_step_hours / 1000  # kWh
                total_energy += energy
                operating_hours += time_step_hours
                peak_power = max(peak_power, power)

        capacity_factor = (
            total_energy / (self.pmax_stc / 1000 * len(irradiance_profile) * time_step_hours)
            if self.pmax_stc > 0 else 0
        )

        return {
            'annual_energy_kwh': total_energy,
            'specific_yield_kwh_kwp': total_energy / (self.pmax_stc / 1000) if self.pmax_stc > 0 else 0,
            'operating_hours': operating_hours,
            'capacity_factor': capacity_factor,
            'peak_power_w': peak_power
        }

    def generate_efficiency_curve(
        self,
        temperature: float = 25.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate efficiency vs irradiance curve.

        Args:
            temperature: Module temperature (°C)

        Returns:
            (irradiance array, efficiency array)
        """
        irradiances = np.linspace(100, 1100, 50)
        efficiencies = []

        for irr in irradiances:
            power = self.get_power(irr, temperature)
            eff = power / (irr * self.module_area) * 100 if self.module_area > 0 else 0
            efficiencies.append(eff)

        return irradiances, np.array(efficiencies)

    def generate_temperature_derating_curve(
        self,
        irradiance: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate power derating vs temperature curve.

        Args:
            irradiance: Irradiance level (W/m²)

        Returns:
            (temperature array, relative power array)
        """
        temperatures = np.linspace(0, 80, 50)
        power_stc = self.get_power(irradiance, 25.0)

        relative_power = []
        for temp in temperatures:
            power = self.get_power(irradiance, temp)
            rel = power / power_stc if power_stc > 0 else 0
            relative_power.append(rel)

        return temperatures, np.array(relative_power)


def create_sample_power_matrix(pmax_stc: float, temp_coeff: float = -0.35) -> PowerMatrix:
    """Create a sample power matrix for testing.

    Args:
        pmax_stc: STC power rating (W)
        temp_coeff: Temperature coefficient (%/°C)

    Returns:
        PowerMatrix with estimated values
    """
    matrix = PowerMatrix()
    matrix.pmax_stc = pmax_stc

    # Typical efficiency curve parameters
    # Low-light efficiency typically drops
    efficiency_factors = {
        100: 0.85,
        200: 0.92,
        400: 0.97,
        600: 0.99,
        800: 1.00,
        1000: 1.00,
        1100: 0.995
    }

    for irr in matrix.irradiance_levels:
        for temp in matrix.temperature_levels:
            # Base power at this irradiance
            p_base = pmax_stc * (irr / 1000) * efficiency_factors.get(irr, 1.0)

            # Temperature correction
            temp_factor = 1 + (temp_coeff / 100) * (temp - 25)
            pmax = p_base * temp_factor

            # Estimate other parameters
            voc_stc = 50.0  # Assume 50V Voc
            isc_stc = pmax_stc / (voc_stc * 0.78)  # Assume FF = 0.78

            point = PowerMatrixPoint(
                irradiance=irr,
                temperature=temp,
                pmax=max(0, pmax),
                voc=voc_stc * (1 - 0.003 * (temp - 25)) * np.log(irr / 1000 + 1) / np.log(2) + voc_stc * 0.5,
                isc=isc_stc * (irr / 1000) * (1 + 0.0005 * (temp - 25)),
                vmpp=voc_stc * 0.8 * (1 - 0.004 * (temp - 25)),
                impp=isc_stc * 0.95 * (irr / 1000),
                ff=0.78 * (1 - 0.001 * (temp - 25))
            )
            matrix.add_point(point)

    return matrix


def calculate_energy_rating(
    pmax_stc: float,
    module_area: float,
    temp_coeff: float = -0.35,
    climate: ClimateProfile = ClimateProfile.TEMPERATE_CONTINENTAL
) -> EnergyRatingResult:
    """Convenience function for quick energy rating calculation.

    Args:
        pmax_stc: STC power rating (W)
        module_area: Module area (m²)
        temp_coeff: Temperature coefficient (%/°C)
        climate: Climate profile

    Returns:
        EnergyRatingResult
    """
    matrix = create_sample_power_matrix(pmax_stc, temp_coeff)
    calculator = EnergyRatingCalculator(matrix, module_area, pmax_stc, temp_coeff)
    return calculator.calculate_cser(climate)
