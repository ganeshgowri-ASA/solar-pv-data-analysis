"""Solar PV Analysis Module.

Comprehensive analysis tools for photovoltaic testing per IEC standards.

Modules:
- iv_curve: I-V curve parameter extraction (Isc, Voc, Pmax, FF, Rs, Rsh)
- corrections: IEC 60891 procedures 1-4 for STC translation
- stabilization: Irradiance and temperature stabilization gates
- gates_calculation: Pass/fail determination with tolerance gates
- spectral_mismatch: Spectral mismatch factor (M) per IEC 60904-7
- classifier: Solar simulator ABC+ classification per IEC 60904-9
- temperature_coeff: Temperature coefficient extraction (α, β, γ, κ)
- uncertainty: GUM uncertainty propagation and budgets
- hotspot_detection: Cell hotspot and weak cell detection with heatmaps
- energy_rating: IEC 61853 energy rating (CSER, CSPR, power matrix)
- iam_analysis: Incidence Angle Modifier curve fitting and losses
- bifaciality: IEC TS 60904-1-2 bifaciality factor and gain

IEC Standards Implementation:
- IEC 60904-1: I-V curve characterization
- IEC 60904-7: Spectral mismatch correction
- IEC 60904-9: Solar simulator classification
- IEC 60904-10: Temperature coefficient measurement
- IEC 60891: Temperature and irradiance corrections
- IEC 61215: Design qualification
- IEC 61853: Energy rating and power matrix
- IEC TS 60904-1-2: Bifacial module testing
"""

from .iv_curve import IVCurveAnalyzer
from .corrections import (
    CorrectionProcedure1,
    CorrectionProcedure2,
    CorrectionProcedure3,
    CorrectionProcedure4,
    STCConditions,
)
from .stabilization import (
    StabilizationMonitor,
    StabilizationStatus,
    StabilizationResult,
    check_irradiance_gate,
    check_temperature_gate,
    calculate_non_uniformity,
    calculate_temporal_instability,
    validate_measurement_conditions,
    EnvironmentalMonitor,
)
from .gates_calculation import (
    ToleranceGate,
    GateSet,
    GateResult,
    DeviationSeverity,
    ConformityDecision,
    DeviationAnalyzer,
    DeviationFlag,
    create_power_tolerance_gates,
    create_iv_parameter_gates,
    create_degradation_gates,
    generate_conformity_statement,
    calculate_cpk,
    calculate_yield,
    power_bin_classification,
)
from .spectral_mismatch import (
    SpectralData,
    calculate_mismatch_factor,
    calculate_mismatch_correction,
    calculate_band_ratios,
    classify_spectral_match,
    estimate_m_factor_simplified,
    generate_spectral_report,
    MismatchUncertainty,
    get_am15g_reference,
    get_typical_sr_csi,
)
from .classifier import (
    SimulatorClassifier,
    NonUniformityMeasurement,
    TemporalStabilityMeasurement,
    SpectralMatchMeasurement,
    ClassificationResult,
    MeasurementType,
    quick_classify,
    parse_classification_string,
    compare_classifications,
)
from .temperature_coeff import (
    TemperatureCoefficientExtractor,
    TemperatureDataPoint,
    TemperatureCoefficientResult,
    extract_rs_from_iv_curves,
    extract_rs_from_two_curves,
    get_typical_coefficients,
    compare_to_typical,
)
from .uncertainty import (
    UncertaintyBudget,
    UncertaintyComponent,
    UncertaintyType,
    UncertaintyResult,
    Distribution,
    CorrelationType,
    create_pmax_uncertainty_budget,
    create_isc_uncertainty_budget,
    create_voc_uncertainty_budget,
    propagate_uncertainty_product,
    propagate_uncertainty_ratio,
    propagate_uncertainty_sum,
    calculate_fill_factor_uncertainty,
)
from .hotspot_detection import (
    HotspotDetector,
    HotspotResult,
    HotspotAnalysisResult,
    StringMismatchResult,
    CellConfiguration,
    CellLayout,
    CellStatus,
    detect_hotspots,
    generate_cell_heatmap,
    analyze_string_currents,
)
from .energy_rating import (
    EnergyRatingCalculator,
    EnergyRatingResult,
    PowerMatrix,
    PowerMatrixPoint,
    ClimateProfile,
    ClimateData,
    CLIMATE_PROFILES,
    create_sample_power_matrix,
    calculate_energy_rating,
)
from .iam_analysis import (
    IAMAnalyzer,
    IAMModel,
    IAMParameters,
    IAMResult,
    IAMCurveFitResult,
    calculate_iam,
    fit_iam_data,
    get_typical_iam_parameters,
)
from .bifaciality import (
    BifacialAnalyzer,
    BifacialIVData,
    BifacialityResult,
    BifacialGainResult,
    MeasurementSide,
    MountingConfiguration,
    calculate_bifaciality_factor,
    estimate_bifacial_gain,
    get_typical_bifaciality,
    get_ground_albedo,
)

__all__ = [
    # I-V Curve Analysis
    "IVCurveAnalyzer",
    # Corrections
    "CorrectionProcedure1",
    "CorrectionProcedure2",
    "CorrectionProcedure3",
    "CorrectionProcedure4",
    "STCConditions",
    # Stabilization
    "StabilizationMonitor",
    "StabilizationStatus",
    "StabilizationResult",
    "check_irradiance_gate",
    "check_temperature_gate",
    "calculate_non_uniformity",
    "calculate_temporal_instability",
    "validate_measurement_conditions",
    "EnvironmentalMonitor",
    # Gates
    "ToleranceGate",
    "GateSet",
    "GateResult",
    "DeviationSeverity",
    "ConformityDecision",
    "DeviationAnalyzer",
    "DeviationFlag",
    "create_power_tolerance_gates",
    "create_iv_parameter_gates",
    "create_degradation_gates",
    "generate_conformity_statement",
    "calculate_cpk",
    "calculate_yield",
    "power_bin_classification",
    # Spectral Mismatch
    "SpectralData",
    "calculate_mismatch_factor",
    "calculate_mismatch_correction",
    "calculate_band_ratios",
    "classify_spectral_match",
    "estimate_m_factor_simplified",
    "generate_spectral_report",
    "MismatchUncertainty",
    "get_am15g_reference",
    "get_typical_sr_csi",
    # Classifier
    "SimulatorClassifier",
    "NonUniformityMeasurement",
    "TemporalStabilityMeasurement",
    "SpectralMatchMeasurement",
    "ClassificationResult",
    "MeasurementType",
    "quick_classify",
    "parse_classification_string",
    "compare_classifications",
    # Temperature Coefficients
    "TemperatureCoefficientExtractor",
    "TemperatureDataPoint",
    "TemperatureCoefficientResult",
    "extract_rs_from_iv_curves",
    "extract_rs_from_two_curves",
    "get_typical_coefficients",
    "compare_to_typical",
    # Uncertainty
    "UncertaintyBudget",
    "UncertaintyComponent",
    "UncertaintyType",
    "UncertaintyResult",
    "Distribution",
    "CorrelationType",
    "create_pmax_uncertainty_budget",
    "create_isc_uncertainty_budget",
    "create_voc_uncertainty_budget",
    "propagate_uncertainty_product",
    "propagate_uncertainty_ratio",
    "propagate_uncertainty_sum",
    "calculate_fill_factor_uncertainty",
    # Hotspot Detection
    "HotspotDetector",
    "HotspotResult",
    "HotspotAnalysisResult",
    "StringMismatchResult",
    "CellConfiguration",
    "CellLayout",
    "CellStatus",
    "detect_hotspots",
    "generate_cell_heatmap",
    "analyze_string_currents",
    # Energy Rating (IEC 61853)
    "EnergyRatingCalculator",
    "EnergyRatingResult",
    "PowerMatrix",
    "PowerMatrixPoint",
    "ClimateProfile",
    "ClimateData",
    "CLIMATE_PROFILES",
    "create_sample_power_matrix",
    "calculate_energy_rating",
    # IAM Analysis
    "IAMAnalyzer",
    "IAMModel",
    "IAMParameters",
    "IAMResult",
    "IAMCurveFitResult",
    "calculate_iam",
    "fit_iam_data",
    "get_typical_iam_parameters",
    # Bifaciality (IEC TS 60904-1-2)
    "BifacialAnalyzer",
    "BifacialIVData",
    "BifacialityResult",
    "BifacialGainResult",
    "MeasurementSide",
    "MountingConfiguration",
    "calculate_bifaciality_factor",
    "estimate_bifacial_gain",
    "get_typical_bifaciality",
    "get_ground_albedo",
]
