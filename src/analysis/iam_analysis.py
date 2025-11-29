"""Incidence Angle Modifier (IAM) Analysis Module.

Implements IAM curve fitting and angular loss calculations for PV modules.

IAM Models:
- ASHRAE model: 1 - b₀(1/cos(θ) - 1)
- Physical model: Based on Fresnel reflections
- Martin-Ruiz model: exp(-cos(θ)/aᵣ) * (1 - exp(-1/aᵣ))⁻¹

References:
- IEC 61853-2: Spectral responsivity, incidence angle and module operating temperature
- De Soto et al. (2006): Improvement and validation of a model for PV array performance
- Martin & Ruiz (2001): Calculation of the PV modules angular losses
"""

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import quad
from typing import Dict, List, Optional, Tuple, NamedTuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class IAMModel(Enum):
    """Available IAM models."""
    ASHRAE = "ashrae"
    PHYSICAL = "physical"
    MARTIN_RUIZ = "martin_ruiz"
    SANDIA = "sandia"
    IEC61853 = "iec61853"


@dataclass
class IAMParameters:
    """IAM model parameters."""
    model: IAMModel
    b0: Optional[float] = None          # ASHRAE parameter
    ar: Optional[float] = None          # Martin-Ruiz angular losses coefficient
    n_glass: Optional[float] = None     # Glass refractive index (physical model)
    n_arc: Optional[float] = None       # ARC refractive index (physical model)
    theta_ref: float = 0.0              # Reference angle (degrees)
    coefficients: Optional[List[float]] = None  # Polynomial coefficients


class IAMResult(NamedTuple):
    """IAM calculation result."""
    angle: float              # Incidence angle (degrees)
    iam: float               # IAM value (0-1)
    angular_loss: float      # Angular loss (1 - IAM) as percentage
    model: str               # Model used


@dataclass
class IAMCurveFitResult:
    """Result of IAM curve fitting."""
    model: IAMModel
    parameters: IAMParameters
    fitted_iam: np.ndarray
    angles: np.ndarray
    residuals: np.ndarray
    rmse: float
    r_squared: float
    iam_60: float  # IAM at 60 degrees (key metric)


class IAMAnalyzer:
    """Analyzer for Incidence Angle Modifier curves."""

    # Typical IAM parameters for different technologies
    TYPICAL_PARAMETERS = {
        'glass_cell_glass': {'b0': 0.05, 'ar': 0.16, 'n_glass': 1.526},
        'glass_cell_polymer': {'b0': 0.05, 'ar': 0.17, 'n_glass': 1.526},
        'polymer_thinfilm_steel': {'b0': 0.05, 'ar': 0.20, 'n_glass': 1.50},
        'cdte': {'b0': 0.046, 'ar': 0.15, 'n_glass': 1.526},
        'cigs': {'b0': 0.048, 'ar': 0.16, 'n_glass': 1.526},
    }

    def __init__(self, model: IAMModel = IAMModel.ASHRAE):
        """Initialize IAM analyzer.

        Args:
            model: IAM model to use
        """
        self.model = model
        self._fit_result = None

    @staticmethod
    def ashrae_iam(theta: np.ndarray, b0: float) -> np.ndarray:
        """Calculate IAM using ASHRAE model.

        IAM = 1 - b₀(1/cos(θ) - 1)

        Args:
            theta: Incidence angle in degrees
            b0: ASHRAE parameter (typically 0.04-0.06)

        Returns:
            IAM values
        """
        theta_rad = np.radians(np.asarray(theta))
        cos_theta = np.cos(theta_rad)

        # Handle edge cases
        cos_theta = np.clip(cos_theta, 0.001, 1.0)

        iam = 1 - b0 * (1 / cos_theta - 1)
        return np.clip(iam, 0, 1)

    @staticmethod
    def physical_iam(
        theta: np.ndarray,
        n_glass: float = 1.526,
        n_arc: float = 1.3,
        k_glass: float = 4.0,
        l_glass: float = 0.002
    ) -> np.ndarray:
        """Calculate IAM using physical (Fresnel) model.

        Based on reflection and absorption through glass layers.

        Args:
            theta: Incidence angle in degrees
            n_glass: Glass refractive index
            n_arc: Anti-reflective coating refractive index
            k_glass: Glass extinction coefficient (1/m)
            l_glass: Glass thickness (m)

        Returns:
            IAM values
        """
        theta_rad = np.radians(np.asarray(theta))
        theta = np.asarray(theta)

        # Calculate refraction angles (Snell's law)
        sin_theta = np.sin(theta_rad)
        sin_theta_arc = sin_theta / n_arc
        sin_theta_glass = sin_theta / n_glass

        # Clip to valid range
        sin_theta_arc = np.clip(sin_theta_arc, -1, 1)
        sin_theta_glass = np.clip(sin_theta_glass, -1, 1)

        theta_arc = np.arcsin(sin_theta_arc)
        theta_glass = np.arcsin(sin_theta_glass)

        # Fresnel reflectance (averaged s and p polarization)
        def fresnel_reflectance(theta1, theta2):
            cos1, cos2 = np.cos(theta1), np.cos(theta2)
            sin1, sin2 = np.sin(theta1), np.sin(theta2)

            # s-polarization
            rs = ((cos1 - n_glass * cos2) / (cos1 + n_glass * cos2)) ** 2

            # p-polarization
            rp = ((n_glass * cos1 - cos2) / (n_glass * cos1 + cos2)) ** 2

            return (rs + rp) / 2

        # Reflectance at air-glass interface
        r = fresnel_reflectance(theta_rad, theta_glass)
        r0 = fresnel_reflectance(np.zeros_like(theta_rad), np.zeros_like(theta_glass))

        # Absorption through glass
        path_length = l_glass / np.cos(theta_glass)
        path_length = np.clip(path_length, 0, 0.1)  # Limit path length
        tau_abs = np.exp(-k_glass * path_length)
        tau_abs_0 = np.exp(-k_glass * l_glass)

        # Transmittance
        tau = (1 - r) * tau_abs
        tau_0 = (1 - r0) * tau_abs_0

        # IAM
        iam = tau / tau_0
        return np.clip(iam, 0, 1)

    @staticmethod
    def martin_ruiz_iam(theta: np.ndarray, ar: float) -> np.ndarray:
        """Calculate IAM using Martin-Ruiz model.

        IAM = exp(-cos(θ)/aᵣ) * (1 - exp(-1/aᵣ))⁻¹ for θ < 90°

        Args:
            theta: Incidence angle in degrees
            ar: Angular losses coefficient (typically 0.15-0.20)

        Returns:
            IAM values
        """
        theta_rad = np.radians(np.asarray(theta))
        cos_theta = np.cos(theta_rad)

        # Handle edge cases
        cos_theta = np.clip(cos_theta, 0, 1)

        # Martin-Ruiz equation
        if ar > 0:
            numerator = np.exp(-cos_theta / ar)
            denominator = 1 - np.exp(-1 / ar)
            iam = 1 - numerator / denominator
        else:
            iam = np.ones_like(theta_rad)

        # Set IAM to 0 at 90 degrees
        iam = np.where(np.abs(theta) >= 90, 0, iam)

        return np.clip(iam, 0, 1)

    @staticmethod
    def sandia_iam(theta: np.ndarray, b0: float, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
        """Calculate IAM using Sandia polynomial model.

        IAM = 1 + b0*θ + b1*θ² + b2*θ³ + b3*θ⁴ + b4*θ⁵

        Args:
            theta: Incidence angle in degrees
            b0-b5: Polynomial coefficients

        Returns:
            IAM values
        """
        theta = np.asarray(theta)
        iam = (1 + b0 * theta + b1 * theta**2 + b2 * theta**3 +
               b3 * theta**4 + b4 * theta**5)
        return np.clip(iam, 0, 1)

    def calculate_iam(
        self,
        theta: float,
        parameters: Optional[IAMParameters] = None
    ) -> IAMResult:
        """Calculate IAM at specific angle.

        Args:
            theta: Incidence angle in degrees
            parameters: Model parameters (uses defaults if None)

        Returns:
            IAMResult
        """
        if parameters is None:
            parameters = self._get_default_parameters()

        theta = np.atleast_1d(theta)

        if self.model == IAMModel.ASHRAE:
            iam = self.ashrae_iam(theta, parameters.b0 or 0.05)
        elif self.model == IAMModel.PHYSICAL:
            iam = self.physical_iam(
                theta,
                parameters.n_glass or 1.526,
                parameters.n_arc or 1.3
            )
        elif self.model == IAMModel.MARTIN_RUIZ:
            iam = self.martin_ruiz_iam(theta, parameters.ar or 0.16)
        else:
            iam = self.ashrae_iam(theta, 0.05)

        iam_val = float(iam[0]) if len(iam) == 1 else iam
        angular_loss = (1 - iam_val) * 100 if isinstance(iam_val, float) else (1 - iam) * 100

        return IAMResult(
            angle=float(theta[0]) if len(theta) == 1 else theta,
            iam=iam_val,
            angular_loss=angular_loss,
            model=self.model.value
        )

    def _get_default_parameters(self) -> IAMParameters:
        """Get default parameters for current model."""
        typical = self.TYPICAL_PARAMETERS['glass_cell_glass']
        return IAMParameters(
            model=self.model,
            b0=typical['b0'],
            ar=typical['ar'],
            n_glass=typical['n_glass']
        )

    def fit_iam_curve(
        self,
        angles: np.ndarray,
        measured_iam: np.ndarray,
        model: Optional[IAMModel] = None
    ) -> IAMCurveFitResult:
        """Fit IAM model to measured data.

        Args:
            angles: Measured angles in degrees
            measured_iam: Measured IAM values
            model: Model to fit (uses instance model if None)

        Returns:
            IAMCurveFitResult with fitted parameters
        """
        if model is None:
            model = self.model

        angles = np.asarray(angles)
        measured_iam = np.asarray(measured_iam)

        # Normalize IAM if needed
        if measured_iam[0] > 1:
            measured_iam = measured_iam / measured_iam[0]

        if model == IAMModel.ASHRAE:
            # Fit ASHRAE model
            try:
                popt, _ = curve_fit(
                    self.ashrae_iam,
                    angles,
                    measured_iam,
                    p0=[0.05],
                    bounds=(0, 0.2)
                )
                b0 = popt[0]
            except Exception:
                b0 = 0.05

            params = IAMParameters(model=model, b0=b0)
            fitted = self.ashrae_iam(angles, b0)

        elif model == IAMModel.MARTIN_RUIZ:
            # Fit Martin-Ruiz model
            try:
                popt, _ = curve_fit(
                    self.martin_ruiz_iam,
                    angles,
                    measured_iam,
                    p0=[0.16],
                    bounds=(0.01, 0.5)
                )
                ar = popt[0]
            except Exception:
                ar = 0.16

            params = IAMParameters(model=model, ar=ar)
            fitted = self.martin_ruiz_iam(angles, ar)

        elif model == IAMModel.PHYSICAL:
            # Fit physical model (n_glass)
            def fit_func(theta, n_glass):
                return self.physical_iam(theta, n_glass)

            try:
                popt, _ = curve_fit(
                    fit_func,
                    angles,
                    measured_iam,
                    p0=[1.526],
                    bounds=(1.3, 1.8)
                )
                n_glass = popt[0]
            except Exception:
                n_glass = 1.526

            params = IAMParameters(model=model, n_glass=n_glass)
            fitted = self.physical_iam(angles, n_glass)

        else:
            params = self._get_default_parameters()
            fitted = self.ashrae_iam(angles, params.b0)

        # Calculate fit statistics
        residuals = measured_iam - fitted
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((measured_iam - np.mean(measured_iam)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # IAM at 60 degrees
        if model == IAMModel.ASHRAE:
            iam_60 = float(self.ashrae_iam(np.array([60]), params.b0)[0])
        elif model == IAMModel.MARTIN_RUIZ:
            iam_60 = float(self.martin_ruiz_iam(np.array([60]), params.ar)[0])
        else:
            iam_60 = float(self.physical_iam(np.array([60]), params.n_glass or 1.526)[0])

        self._fit_result = IAMCurveFitResult(
            model=model,
            parameters=params,
            fitted_iam=fitted,
            angles=angles,
            residuals=residuals,
            rmse=rmse,
            r_squared=r_squared,
            iam_60=iam_60
        )

        return self._fit_result

    def calculate_annual_angular_loss(
        self,
        latitude: float,
        tilt: float = 0.0,
        azimuth: float = 180.0,
        parameters: Optional[IAMParameters] = None
    ) -> Dict[str, float]:
        """Calculate annual angular losses for a given installation.

        Args:
            latitude: Site latitude in degrees
            tilt: Array tilt angle in degrees
            azimuth: Array azimuth (180 = south)
            parameters: IAM parameters

        Returns:
            Dictionary with loss metrics
        """
        if parameters is None:
            parameters = self._get_default_parameters()

        # Simplified calculation using representative angles
        # Full calculation would integrate over all sun positions

        # Representative incidence angles through the day/year
        representative_angles = np.array([0, 15, 30, 45, 60, 75, 85])
        weights = np.array([0.1, 0.15, 0.25, 0.25, 0.15, 0.08, 0.02])  # Typical distribution

        # Calculate weighted average IAM
        if self.model == IAMModel.ASHRAE:
            iam_values = self.ashrae_iam(representative_angles, parameters.b0 or 0.05)
        elif self.model == IAMModel.MARTIN_RUIZ:
            iam_values = self.martin_ruiz_iam(representative_angles, parameters.ar or 0.16)
        else:
            iam_values = self.physical_iam(representative_angles, parameters.n_glass or 1.526)

        avg_iam = np.sum(iam_values * weights)
        angular_loss = (1 - avg_iam) * 100

        # Estimate impact of tilt
        # Higher tilt generally reduces angular losses
        tilt_factor = 1 - 0.005 * tilt
        adjusted_loss = angular_loss * tilt_factor

        return {
            'average_iam': avg_iam,
            'annual_angular_loss_percent': angular_loss,
            'adjusted_loss_percent': adjusted_loss,
            'iam_at_0_deg': float(iam_values[0]),
            'iam_at_60_deg': float(iam_values[4]),
            'latitude': latitude,
            'tilt': tilt,
            'azimuth': azimuth
        }

    def get_iam_curve(
        self,
        parameters: Optional[IAMParameters] = None,
        angles: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full IAM curve.

        Args:
            parameters: IAM parameters
            angles: Angle array (default 0-90 degrees)

        Returns:
            (angles, iam_values)
        """
        if parameters is None:
            parameters = self._get_default_parameters()

        if angles is None:
            angles = np.linspace(0, 90, 91)

        if self.model == IAMModel.ASHRAE:
            iam = self.ashrae_iam(angles, parameters.b0 or 0.05)
        elif self.model == IAMModel.MARTIN_RUIZ:
            iam = self.martin_ruiz_iam(angles, parameters.ar or 0.16)
        elif self.model == IAMModel.PHYSICAL:
            iam = self.physical_iam(angles, parameters.n_glass or 1.526)
        else:
            iam = self.ashrae_iam(angles, 0.05)

        return angles, iam

    def compare_models(
        self,
        angles: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Compare different IAM models.

        Args:
            angles: Angle array (default 0-90 degrees)

        Returns:
            Dictionary with model name -> IAM values
        """
        if angles is None:
            angles = np.linspace(0, 90, 91)

        return {
            'angles': angles,
            'ashrae_b0_05': self.ashrae_iam(angles, 0.05),
            'ashrae_b0_04': self.ashrae_iam(angles, 0.04),
            'martin_ruiz_ar_16': self.martin_ruiz_iam(angles, 0.16),
            'martin_ruiz_ar_20': self.martin_ruiz_iam(angles, 0.20),
            'physical_n_1526': self.physical_iam(angles, 1.526),
            'physical_n_150': self.physical_iam(angles, 1.50)
        }


def calculate_iam(
    theta: float,
    model: IAMModel = IAMModel.ASHRAE,
    b0: float = 0.05,
    ar: float = 0.16
) -> float:
    """Convenience function to calculate IAM at specific angle.

    Args:
        theta: Incidence angle in degrees
        model: IAM model to use
        b0: ASHRAE parameter
        ar: Martin-Ruiz parameter

    Returns:
        IAM value
    """
    analyzer = IAMAnalyzer(model)
    params = IAMParameters(model=model, b0=b0, ar=ar)
    result = analyzer.calculate_iam(theta, params)
    return result.iam


def fit_iam_data(
    angles: np.ndarray,
    measured_iam: np.ndarray,
    model: IAMModel = IAMModel.ASHRAE
) -> IAMCurveFitResult:
    """Convenience function to fit IAM data.

    Args:
        angles: Measured angles in degrees
        measured_iam: Measured IAM values
        model: Model to fit

    Returns:
        IAMCurveFitResult
    """
    analyzer = IAMAnalyzer(model)
    return analyzer.fit_iam_curve(angles, measured_iam, model)


def get_typical_iam_parameters(technology: str = 'glass_cell_glass') -> Dict[str, float]:
    """Get typical IAM parameters for a technology.

    Args:
        technology: Module technology type

    Returns:
        Dictionary with typical parameters
    """
    return IAMAnalyzer.TYPICAL_PARAMETERS.get(
        technology,
        IAMAnalyzer.TYPICAL_PARAMETERS['glass_cell_glass']
    )
