"""Solar Simulator Classification Module (IEC 60904-9:2020).

Implements comprehensive classification of solar simulators according to
IEC 60904-9:2020 requirements including:
- Spectral match (6 wavelength bands)
- Spatial non-uniformity of irradiance
- Short-term temporal instability (STI)
- Long-term temporal instability (LTI)

Classification grades: A+, A, B, C
Overall classification expressed as three letters (e.g., "AAA", "A+BA")
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.iec_standards import (
    SimulatorClass,
    SPECTRAL_MATCH_CRITERIA,
    NON_UNIFORMITY_CRITERIA,
    STI_CRITERIA,
    LTI_CRITERIA,
    WAVELENGTH_BANDS,
    IEC60904_9_Classification,
    classify_simulator
)


class MeasurementType(Enum):
    """Type of simulator measurement."""
    SPECTRAL = "spectral"
    NON_UNIFORMITY = "non_uniformity"
    TEMPORAL_STI = "temporal_sti"
    TEMPORAL_LTI = "temporal_lti"


class ClassificationResult(NamedTuple):
    """Result of a single classification measurement."""
    measurement_type: MeasurementType
    value: float
    unit: str
    classification: SimulatorClass
    threshold_used: float
    margin_to_next_class: float


@dataclass
class NonUniformityMeasurement:
    """Spatial non-uniformity measurement data."""
    # Grid measurements (irradiance values at each position)
    grid_data: np.ndarray  # 2D array of irradiance values
    grid_rows: int = 0
    grid_cols: int = 0

    # Measurement metadata
    measurement_date: Optional[datetime] = None
    test_area_width_mm: float = 0.0
    test_area_height_mm: float = 0.0
    grid_spacing_mm: float = 0.0

    def __post_init__(self):
        """Initialize grid dimensions."""
        if self.grid_data is not None:
            self.grid_data = np.array(self.grid_data)
            if len(self.grid_data.shape) == 2:
                self.grid_rows, self.grid_cols = self.grid_data.shape
            elif len(self.grid_data.shape) == 1:
                self.grid_rows = 1
                self.grid_cols = len(self.grid_data)

    def calculate_non_uniformity(self) -> float:
        """Calculate non-uniformity per IEC 60904-9.

        Non-uniformity = ((max - min) / (max + min)) × 100%

        Returns:
            Non-uniformity percentage
        """
        if self.grid_data is None or self.grid_data.size == 0:
            return 0.0

        flat_data = self.grid_data.flatten()
        max_val = np.max(flat_data)
        min_val = np.min(flat_data)

        if (max_val + min_val) == 0:
            return 0.0

        return ((max_val - min_val) / (max_val + min_val)) * 100.0

    def calculate_cv(self) -> float:
        """Calculate coefficient of variation.

        Returns:
            CV as percentage
        """
        if self.grid_data is None or self.grid_data.size == 0:
            return 0.0

        flat_data = self.grid_data.flatten()
        mean = np.mean(flat_data)
        std = np.std(flat_data)

        if mean == 0:
            return 0.0

        return (std / mean) * 100.0

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of grid data."""
        if self.grid_data is None or self.grid_data.size == 0:
            return {}

        flat_data = self.grid_data.flatten()

        return {
            "min": float(np.min(flat_data)),
            "max": float(np.max(flat_data)),
            "mean": float(np.mean(flat_data)),
            "std": float(np.std(flat_data)),
            "cv_percent": self.calculate_cv(),
            "non_uniformity_percent": self.calculate_non_uniformity(),
            "n_measurements": len(flat_data)
        }

    def classify(self) -> ClassificationResult:
        """Classify non-uniformity per IEC 60904-9.

        Returns:
            ClassificationResult with classification details
        """
        nu = self.calculate_non_uniformity()
        classification = NON_UNIFORMITY_CRITERIA.classify(nu)

        # Calculate margin to next class
        thresholds = {
            SimulatorClass.A_PLUS: NON_UNIFORMITY_CRITERIA.a_plus_max,
            SimulatorClass.A: NON_UNIFORMITY_CRITERIA.a_max,
            SimulatorClass.B: NON_UNIFORMITY_CRITERIA.b_max,
            SimulatorClass.C: NON_UNIFORMITY_CRITERIA.c_max
        }

        threshold = thresholds[classification]
        margin = threshold - nu

        return ClassificationResult(
            measurement_type=MeasurementType.NON_UNIFORMITY,
            value=nu,
            unit="%",
            classification=classification,
            threshold_used=threshold,
            margin_to_next_class=margin
        )


@dataclass
class TemporalStabilityMeasurement:
    """Temporal stability measurement data."""
    # Time series of irradiance measurements
    irradiance_values: np.ndarray
    time_stamps_ms: Optional[np.ndarray] = None

    # Measurement metadata
    measurement_type: str = "sti"  # "sti" or "lti"
    measurement_duration_ms: float = 0.0
    sampling_rate_hz: float = 0.0
    measurement_date: Optional[datetime] = None

    def __post_init__(self):
        """Initialize measurement arrays."""
        self.irradiance_values = np.array(self.irradiance_values)
        if self.time_stamps_ms is not None:
            self.time_stamps_ms = np.array(self.time_stamps_ms)

    def calculate_instability(self) -> float:
        """Calculate temporal instability per IEC 60904-9.

        Instability = ((max - min) / (max + min)) × 100%

        Returns:
            Temporal instability percentage
        """
        if len(self.irradiance_values) < 2:
            return 0.0

        max_val = np.max(self.irradiance_values)
        min_val = np.min(self.irradiance_values)

        if (max_val + min_val) == 0:
            return 0.0

        return ((max_val - min_val) / (max_val + min_val)) * 100.0

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of time series."""
        if len(self.irradiance_values) == 0:
            return {}

        return {
            "min": float(np.min(self.irradiance_values)),
            "max": float(np.max(self.irradiance_values)),
            "mean": float(np.mean(self.irradiance_values)),
            "std": float(np.std(self.irradiance_values)),
            "instability_percent": self.calculate_instability(),
            "n_samples": len(self.irradiance_values),
            "duration_ms": self.measurement_duration_ms
        }

    def classify(self) -> ClassificationResult:
        """Classify temporal stability per IEC 60904-9.

        Returns:
            ClassificationResult with classification details
        """
        instability = self.calculate_instability()

        if self.measurement_type.lower() == "sti":
            classification = STI_CRITERIA.classify(instability)
            criteria = STI_CRITERIA
            meas_type = MeasurementType.TEMPORAL_STI
        else:
            classification = LTI_CRITERIA.classify(instability)
            criteria = LTI_CRITERIA
            meas_type = MeasurementType.TEMPORAL_LTI

        # Get threshold for current class
        thresholds = {
            SimulatorClass.A_PLUS: criteria.a_plus_max,
            SimulatorClass.A: criteria.a_max,
            SimulatorClass.B: criteria.b_max,
            SimulatorClass.C: criteria.c_max
        }

        threshold = thresholds[classification]
        margin = threshold - instability

        return ClassificationResult(
            measurement_type=meas_type,
            value=instability,
            unit="%",
            classification=classification,
            threshold_used=threshold,
            margin_to_next_class=margin
        )


@dataclass
class SpectralMatchMeasurement:
    """Spectral match measurement data."""
    # Band ratios (simulator / reference for each band)
    band_ratios: Dict[str, float] = field(default_factory=dict)

    # Optional: raw spectral data
    simulator_spectrum: Optional[np.ndarray] = None
    reference_spectrum: Optional[np.ndarray] = None
    wavelengths_nm: Optional[np.ndarray] = None

    measurement_date: Optional[datetime] = None

    def classify_band(self, band_name: str) -> ClassificationResult:
        """Classify a single wavelength band.

        Args:
            band_name: Name of the wavelength band

        Returns:
            ClassificationResult for the band
        """
        if band_name not in self.band_ratios:
            raise ValueError(f"Unknown band: {band_name}")

        ratio = self.band_ratios[band_name]

        try:
            classification = SPECTRAL_MATCH_CRITERIA.classify(ratio)
        except ValueError:
            classification = SimulatorClass.C

        # Get threshold for current class
        if classification == SimulatorClass.A_PLUS:
            threshold = SPECTRAL_MATCH_CRITERIA.a_plus_max
            margin = min(ratio - SPECTRAL_MATCH_CRITERIA.a_plus_min,
                        SPECTRAL_MATCH_CRITERIA.a_plus_max - ratio)
        elif classification == SimulatorClass.A:
            threshold = SPECTRAL_MATCH_CRITERIA.a_max
            margin = min(ratio - SPECTRAL_MATCH_CRITERIA.a_min,
                        SPECTRAL_MATCH_CRITERIA.a_max - ratio)
        elif classification == SimulatorClass.B:
            threshold = SPECTRAL_MATCH_CRITERIA.b_max
            margin = min(ratio - SPECTRAL_MATCH_CRITERIA.b_min,
                        SPECTRAL_MATCH_CRITERIA.b_max - ratio)
        else:
            threshold = SPECTRAL_MATCH_CRITERIA.c_max
            margin = min(ratio - SPECTRAL_MATCH_CRITERIA.c_min,
                        SPECTRAL_MATCH_CRITERIA.c_max - ratio)

        return ClassificationResult(
            measurement_type=MeasurementType.SPECTRAL,
            value=ratio,
            unit="ratio",
            classification=classification,
            threshold_used=threshold,
            margin_to_next_class=margin
        )

    def classify_overall(self) -> Tuple[SimulatorClass, Dict[str, ClassificationResult]]:
        """Classify overall spectral match.

        Returns:
            (overall_class, per_band_results)
        """
        band_results = {}
        class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]

        for band_name in self.band_ratios:
            band_results[band_name] = self.classify_band(band_name)

        if band_results:
            worst_idx = max(
                class_order.index(r.classification)
                for r in band_results.values()
            )
            overall = class_order[worst_idx]
        else:
            overall = SimulatorClass.C

        return (overall, band_results)


@dataclass
class SimulatorClassifier:
    """Complete solar simulator classifier per IEC 60904-9:2020."""

    # Measurement data
    spectral_measurement: Optional[SpectralMatchMeasurement] = None
    non_uniformity_measurement: Optional[NonUniformityMeasurement] = None
    sti_measurement: Optional[TemporalStabilityMeasurement] = None
    lti_measurement: Optional[TemporalStabilityMeasurement] = None

    # Classification results
    _classification: Optional[IEC60904_9_Classification] = None

    def classify(self) -> IEC60904_9_Classification:
        """Perform full classification.

        Returns:
            IEC60904_9_Classification with complete results
        """
        # Get spectral classification
        spectral_ratios = {}
        spectral_class = SimulatorClass.C

        if self.spectral_measurement:
            spectral_class, _ = self.spectral_measurement.classify_overall()
            spectral_ratios = self.spectral_measurement.band_ratios

        # Get non-uniformity classification
        non_uniformity_percent = 0.0
        non_uniformity_class = SimulatorClass.C

        if self.non_uniformity_measurement:
            nu_result = self.non_uniformity_measurement.classify()
            non_uniformity_percent = nu_result.value
            non_uniformity_class = nu_result.classification

        # Get temporal stability classification
        sti_percent = 0.0
        lti_percent = 0.0
        temporal_class = SimulatorClass.C

        if self.sti_measurement:
            sti_result = self.sti_measurement.classify()
            sti_percent = sti_result.value
            sti_class = sti_result.classification
        else:
            sti_class = SimulatorClass.C

        if self.lti_measurement:
            lti_result = self.lti_measurement.classify()
            lti_percent = lti_result.value
            lti_class = lti_result.classification
        else:
            lti_class = SimulatorClass.C

        # Temporal class is worst of STI and LTI
        class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]
        temporal_class = class_order[
            max(class_order.index(sti_class), class_order.index(lti_class))
        ]

        self._classification = IEC60904_9_Classification(
            spectral_class=spectral_class,
            non_uniformity_class=non_uniformity_class,
            temporal_class=temporal_class,
            spectral_ratios=spectral_ratios,
            non_uniformity_percent=non_uniformity_percent,
            sti_percent=sti_percent,
            lti_percent=lti_percent
        )

        return self._classification

    def get_classification_string(self) -> str:
        """Get classification as string (e.g., 'AAA').

        Returns:
            Classification string
        """
        if self._classification is None:
            self.classify()

        return self._classification.classification_string

    def is_suitable_for_stc(self) -> bool:
        """Check if simulator is suitable for STC measurements.

        Class A or better required for all three criteria.

        Returns:
            True if suitable for STC measurements
        """
        if self._classification is None:
            self.classify()

        return self._classification.is_acceptable_for_stc()

    def generate_report(self) -> str:
        """Generate detailed classification report.

        Returns:
            Formatted report string
        """
        if self._classification is None:
            self.classify()

        c = self._classification

        report = f"""
SOLAR SIMULATOR CLASSIFICATION REPORT
=====================================
Classification per IEC 60904-9:2020

OVERALL CLASSIFICATION: {c.classification_string}
{'='*50}

1. SPECTRAL MATCH: Class {c.spectral_class.value}
   --------------------------------------------
"""

        if c.spectral_ratios:
            for band in WAVELENGTH_BANDS:
                if band.name in c.spectral_ratios:
                    ratio = c.spectral_ratios[band.name]
                    try:
                        band_class = SPECTRAL_MATCH_CRITERIA.classify(ratio)
                    except ValueError:
                        band_class = SimulatorClass.C
                    report += f"   {band.name} ({band.range_str} nm): {ratio:.3f} - Class {band_class.value}\n"

        report += f"""
2. SPATIAL NON-UNIFORMITY: Class {c.non_uniformity_class.value}
   --------------------------------------------------
   Non-uniformity: {c.non_uniformity_percent:.2f}%
   Threshold Class A+: ≤{NON_UNIFORMITY_CRITERIA.a_plus_max}%
   Threshold Class A:  ≤{NON_UNIFORMITY_CRITERIA.a_max}%
   Threshold Class B:  ≤{NON_UNIFORMITY_CRITERIA.b_max}%
   Threshold Class C:  ≤{NON_UNIFORMITY_CRITERIA.c_max}%

3. TEMPORAL INSTABILITY: Class {c.temporal_class.value}
   -----------------------------------------------
   Short-term (STI): {c.sti_percent:.3f}%
   Long-term (LTI):  {c.lti_percent:.3f}%

   STI Thresholds:
   - Class A+: ≤{STI_CRITERIA.a_plus_max}%
   - Class A:  ≤{STI_CRITERIA.a_max}%
   - Class B:  ≤{STI_CRITERIA.b_max}%
   - Class C:  ≤{STI_CRITERIA.c_max}%

   LTI Thresholds:
   - Class A+: <{LTI_CRITERIA.a_plus_max}%
   - Class A:  ≤{LTI_CRITERIA.a_max}%
   - Class B:  ≤{LTI_CRITERIA.b_max}%
   - Class C:  ≤{LTI_CRITERIA.c_max}%

SUITABILITY FOR STC MEASUREMENTS: {'YES' if c.is_acceptable_for_stc() else 'NO'}
{'='*50}
"""

        return report


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_classify(
    spectral_ratios: Dict[str, float],
    non_uniformity_percent: float,
    sti_percent: float,
    lti_percent: float
) -> IEC60904_9_Classification:
    """Quick classification from raw values.

    Args:
        spectral_ratios: Dict of band name to ratio
        non_uniformity_percent: Spatial non-uniformity (%)
        sti_percent: Short-term instability (%)
        lti_percent: Long-term instability (%)

    Returns:
        IEC60904_9_Classification
    """
    return classify_simulator(
        spectral_ratios=spectral_ratios,
        non_uniformity_percent=non_uniformity_percent,
        sti_percent=sti_percent,
        lti_percent=lti_percent
    )


def parse_classification_string(classification: str) -> Dict[str, SimulatorClass]:
    """Parse classification string to individual classes.

    Args:
        classification: String like "AAA" or "A+BA"

    Returns:
        Dict with spectral, non_uniformity, temporal classes
    """
    class_map = {
        "A+": SimulatorClass.A_PLUS,
        "A": SimulatorClass.A,
        "B": SimulatorClass.B,
        "C": SimulatorClass.C
    }

    # Parse string (handle A+ which takes 2 chars)
    parts = []
    i = 0
    while i < len(classification):
        if i + 1 < len(classification) and classification[i + 1] == '+':
            parts.append(classification[i:i + 2])
            i += 2
        else:
            parts.append(classification[i])
            i += 1

    result = {}
    if len(parts) >= 1:
        result["spectral"] = class_map.get(parts[0], SimulatorClass.C)
    if len(parts) >= 2:
        result["non_uniformity"] = class_map.get(parts[1], SimulatorClass.C)
    if len(parts) >= 3:
        result["temporal"] = class_map.get(parts[2], SimulatorClass.C)

    return result


def compare_classifications(
    class1: str,
    class2: str
) -> Dict[str, Any]:
    """Compare two classification strings.

    Args:
        class1: First classification (e.g., "AAA")
        class2: Second classification (e.g., "ABA")

    Returns:
        Dict with comparison results
    """
    parsed1 = parse_classification_string(class1)
    parsed2 = parse_classification_string(class2)

    class_order = [SimulatorClass.A_PLUS, SimulatorClass.A, SimulatorClass.B, SimulatorClass.C]

    comparison = {}
    for key in ["spectral", "non_uniformity", "temporal"]:
        if key in parsed1 and key in parsed2:
            idx1 = class_order.index(parsed1[key])
            idx2 = class_order.index(parsed2[key])

            if idx1 < idx2:
                comparison[key] = f"{class1} better"
            elif idx1 > idx2:
                comparison[key] = f"{class2} better"
            else:
                comparison[key] = "equal"

    return {
        "class1": class1,
        "class2": class2,
        "comparison": comparison,
        "overall_better": class1 if sum(1 for v in comparison.values() if "class1" in v.lower() or v == "equal") >= 2 else class2
    }
