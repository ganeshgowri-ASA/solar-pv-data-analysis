"""Hotspot Detection & Cell Analysis Module.

Statistical hotspot detection from cell temperature data with support for
various cell configurations (60-cell, 72-cell, 120-cell, 144-cell half-cut).

Key Features:
- Statistical hotspot detection from cell temperature data
- Good/weak cell identification per cell configuration
- Cell string mismatch detection
- Visual heatmap generation for cell temperatures
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class CellConfiguration(Enum):
    """Standard PV module cell configurations."""
    CELLS_60 = "60-cell"        # 6x10 layout
    CELLS_72 = "72-cell"        # 6x12 layout
    CELLS_120 = "120-cell"      # Half-cut 6x20 layout (2 strings of 60)
    CELLS_144 = "144-cell"      # Half-cut 6x24 layout (2 strings of 72)
    CELLS_96 = "96-cell"        # 8x12 layout (common for some thin-film)
    CELLS_132 = "132-cell"      # Half-cut 6x22 layout
    CUSTOM = "custom"


@dataclass
class CellLayout:
    """Cell layout configuration."""
    rows: int
    cols: int
    n_cells: int
    n_strings: int
    cells_per_string: int
    is_half_cut: bool

    @classmethod
    def from_configuration(cls, config: CellConfiguration) -> 'CellLayout':
        """Create layout from standard configuration."""
        layouts = {
            CellConfiguration.CELLS_60: cls(10, 6, 60, 3, 20, False),
            CellConfiguration.CELLS_72: cls(12, 6, 72, 3, 24, False),
            CellConfiguration.CELLS_120: cls(20, 6, 120, 6, 20, True),
            CellConfiguration.CELLS_144: cls(24, 6, 144, 6, 24, True),
            CellConfiguration.CELLS_96: cls(12, 8, 96, 4, 24, False),
            CellConfiguration.CELLS_132: cls(22, 6, 132, 6, 22, True),
        }
        return layouts.get(config, cls(10, 6, 60, 3, 20, False))


class CellStatus(Enum):
    """Cell health status classification."""
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    HOTSPOT = "hotspot"
    COOL = "cool"
    WEAK = "weak"


@dataclass
class HotspotResult:
    """Result of hotspot detection for a single cell."""
    cell_index: int
    row: int
    col: int
    temperature: float
    delta_t: float  # Temperature above mean
    z_score: float
    status: CellStatus
    string_id: int


@dataclass
class StringMismatchResult:
    """Result of string mismatch analysis."""
    string_id: int
    mean_temp: float
    std_temp: float
    n_cells: int
    n_hotspots: int
    n_weak_cells: int
    current_estimate: Optional[float]
    mismatch_severity: str  # "none", "minor", "moderate", "severe"


class HotspotAnalysisResult(NamedTuple):
    """Complete hotspot analysis results."""
    hotspot_cells: List[HotspotResult]
    weak_cells: List[HotspotResult]
    normal_cells: List[HotspotResult]
    string_analysis: List[StringMismatchResult]
    mean_temperature: float
    std_temperature: float
    max_delta_t: float
    hotspot_count: int
    weak_cell_count: int
    overall_status: str  # "healthy", "minor_issues", "attention_needed", "critical"
    uniformity_index: float  # 0-100, higher is better


class HotspotDetector:
    """Detector for hotspots and weak cells in PV modules."""

    # Temperature thresholds (delta T from mean)
    WARM_THRESHOLD = 5.0      # 5°C above mean
    HOT_THRESHOLD = 10.0      # 10°C above mean
    HOTSPOT_THRESHOLD = 15.0  # 15°C above mean (potential damage risk)
    COOL_THRESHOLD = -5.0     # 5°C below mean
    WEAK_THRESHOLD = -10.0    # 10°C below mean (weak cell indication)

    # Z-score thresholds
    Z_SCORE_WARNING = 2.0
    Z_SCORE_CRITICAL = 3.0

    def __init__(
        self,
        configuration: CellConfiguration = CellConfiguration.CELLS_72,
        custom_layout: Optional[CellLayout] = None,
        warm_threshold: float = 5.0,
        hot_threshold: float = 10.0,
        hotspot_threshold: float = 15.0
    ):
        """Initialize hotspot detector.

        Args:
            configuration: Standard cell configuration
            custom_layout: Custom layout (overrides configuration)
            warm_threshold: Delta T threshold for warm classification
            hot_threshold: Delta T threshold for hot classification
            hotspot_threshold: Delta T threshold for hotspot classification
        """
        if custom_layout:
            self.layout = custom_layout
        else:
            self.layout = CellLayout.from_configuration(configuration)

        self.configuration = configuration
        self.WARM_THRESHOLD = warm_threshold
        self.HOT_THRESHOLD = hot_threshold
        self.HOTSPOT_THRESHOLD = hotspot_threshold

        self._results = None

    def analyze(
        self,
        temperature_data: np.ndarray,
        ambient_temp: Optional[float] = None,
        irradiance: Optional[float] = None
    ) -> HotspotAnalysisResult:
        """Perform complete hotspot analysis.

        Args:
            temperature_data: 2D array of cell temperatures (rows x cols)
                             or 1D array that will be reshaped
            ambient_temp: Ambient temperature (optional, for normalization)
            irradiance: Irradiance level (optional, for current estimation)

        Returns:
            HotspotAnalysisResult with complete analysis
        """
        # Reshape data if needed
        if temperature_data.ndim == 1:
            if len(temperature_data) != self.layout.n_cells:
                raise ValueError(
                    f"Expected {self.layout.n_cells} cells, "
                    f"got {len(temperature_data)}"
                )
            temp_matrix = temperature_data.reshape(
                self.layout.rows, self.layout.cols
            )
        else:
            temp_matrix = temperature_data

        # Validate dimensions
        if temp_matrix.shape != (self.layout.rows, self.layout.cols):
            raise ValueError(
                f"Temperature data shape {temp_matrix.shape} doesn't match "
                f"layout ({self.layout.rows}, {self.layout.cols})"
            )

        # Calculate statistics
        mean_temp = np.mean(temp_matrix)
        std_temp = np.std(temp_matrix)

        # Normalize by ambient if provided
        if ambient_temp is not None:
            delta_t_ambient = temp_matrix - ambient_temp
        else:
            delta_t_ambient = None

        # Analyze each cell
        hotspot_cells = []
        weak_cells = []
        normal_cells = []

        for row in range(self.layout.rows):
            for col in range(self.layout.cols):
                cell_idx = row * self.layout.cols + col
                temp = temp_matrix[row, col]
                delta_t = temp - mean_temp
                z_score = delta_t / std_temp if std_temp > 0 else 0

                # Determine string ID
                string_id = self._get_string_id(row, col)

                # Classify cell status
                status = self._classify_cell(delta_t, z_score)

                result = HotspotResult(
                    cell_index=cell_idx,
                    row=row,
                    col=col,
                    temperature=temp,
                    delta_t=delta_t,
                    z_score=z_score,
                    status=status,
                    string_id=string_id
                )

                if status in [CellStatus.HOT, CellStatus.HOTSPOT]:
                    hotspot_cells.append(result)
                elif status in [CellStatus.COOL, CellStatus.WEAK]:
                    weak_cells.append(result)
                else:
                    normal_cells.append(result)

        # Analyze strings
        string_analysis = self._analyze_strings(temp_matrix, irradiance)

        # Calculate overall metrics
        max_delta_t = np.max(temp_matrix) - mean_temp
        uniformity_index = self._calculate_uniformity_index(temp_matrix)

        # Determine overall status
        overall_status = self._determine_overall_status(
            len(hotspot_cells),
            len(weak_cells),
            max_delta_t,
            uniformity_index
        )

        self._results = HotspotAnalysisResult(
            hotspot_cells=hotspot_cells,
            weak_cells=weak_cells,
            normal_cells=normal_cells,
            string_analysis=string_analysis,
            mean_temperature=mean_temp,
            std_temperature=std_temp,
            max_delta_t=max_delta_t,
            hotspot_count=len(hotspot_cells),
            weak_cell_count=len(weak_cells),
            overall_status=overall_status,
            uniformity_index=uniformity_index
        )

        return self._results

    def _classify_cell(self, delta_t: float, z_score: float) -> CellStatus:
        """Classify cell status based on temperature deviation."""
        # Use both absolute temperature difference and z-score
        if delta_t >= self.HOTSPOT_THRESHOLD or z_score >= self.Z_SCORE_CRITICAL:
            return CellStatus.HOTSPOT
        elif delta_t >= self.HOT_THRESHOLD or z_score >= self.Z_SCORE_WARNING:
            return CellStatus.HOT
        elif delta_t >= self.WARM_THRESHOLD:
            return CellStatus.WARM
        elif delta_t <= self.WEAK_THRESHOLD:
            return CellStatus.WEAK
        elif delta_t <= self.COOL_THRESHOLD:
            return CellStatus.COOL
        else:
            return CellStatus.NORMAL

    def _get_string_id(self, row: int, col: int) -> int:
        """Determine which string a cell belongs to."""
        if self.layout.is_half_cut:
            # Half-cut modules: left and right halves are separate strings
            half = 0 if col < self.layout.cols // 2 else 1
            string_in_half = row // (self.layout.rows // (self.layout.n_strings // 2))
            return half * (self.layout.n_strings // 2) + string_in_half
        else:
            # Standard modules: horizontal strings
            return row // (self.layout.rows // self.layout.n_strings)

    def _analyze_strings(
        self,
        temp_matrix: np.ndarray,
        irradiance: Optional[float]
    ) -> List[StringMismatchResult]:
        """Analyze each string for mismatch conditions."""
        string_results = []

        for string_id in range(self.layout.n_strings):
            # Get cells in this string
            string_temps = []
            for row in range(self.layout.rows):
                for col in range(self.layout.cols):
                    if self._get_string_id(row, col) == string_id:
                        string_temps.append(temp_matrix[row, col])

            string_temps = np.array(string_temps)
            mean_temp = np.mean(string_temps)
            std_temp = np.std(string_temps)

            # Count issues in string
            n_hotspots = np.sum(string_temps > (np.mean(temp_matrix) + self.HOT_THRESHOLD))
            n_weak = np.sum(string_temps < (np.mean(temp_matrix) + self.WEAK_THRESHOLD))

            # Estimate current if irradiance provided
            current_estimate = None
            if irradiance is not None:
                # Rough estimation: current proportional to inverse temperature
                # (higher temp = more resistance = less current)
                current_estimate = irradiance / 1000.0 * (1 - (mean_temp - 25) * 0.004)

            # Determine mismatch severity
            temp_range = np.max(string_temps) - np.min(string_temps)
            if temp_range < 5:
                severity = "none"
            elif temp_range < 10:
                severity = "minor"
            elif temp_range < 20:
                severity = "moderate"
            else:
                severity = "severe"

            string_results.append(StringMismatchResult(
                string_id=string_id,
                mean_temp=mean_temp,
                std_temp=std_temp,
                n_cells=len(string_temps),
                n_hotspots=int(n_hotspots),
                n_weak_cells=int(n_weak),
                current_estimate=current_estimate,
                mismatch_severity=severity
            ))

        return string_results

    def _calculate_uniformity_index(self, temp_matrix: np.ndarray) -> float:
        """Calculate temperature uniformity index (0-100)."""
        # Based on coefficient of variation
        cv = np.std(temp_matrix) / np.mean(temp_matrix) if np.mean(temp_matrix) > 0 else 0
        # Convert to 0-100 scale (lower CV = higher uniformity)
        uniformity = max(0, 100 * (1 - cv * 10))  # Assuming CV < 0.1 is good
        return min(100, uniformity)

    def _determine_overall_status(
        self,
        n_hotspots: int,
        n_weak: int,
        max_delta_t: float,
        uniformity: float
    ) -> str:
        """Determine overall module health status."""
        total_cells = self.layout.n_cells
        hotspot_pct = n_hotspots / total_cells * 100
        weak_pct = n_weak / total_cells * 100

        if hotspot_pct > 5 or max_delta_t > 25 or uniformity < 70:
            return "critical"
        elif hotspot_pct > 2 or weak_pct > 5 or max_delta_t > 15:
            return "attention_needed"
        elif hotspot_pct > 0 or weak_pct > 2 or uniformity < 90:
            return "minor_issues"
        else:
            return "healthy"

    def generate_heatmap_data(
        self,
        temperature_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Generate data for heatmap visualization.

        Args:
            temperature_data: Cell temperature data

        Returns:
            Tuple of (processed matrix, metadata dict)
        """
        # Reshape if needed
        if temperature_data.ndim == 1:
            temp_matrix = temperature_data.reshape(
                self.layout.rows, self.layout.cols
            )
        else:
            temp_matrix = temperature_data

        metadata = {
            'min_temp': float(np.min(temp_matrix)),
            'max_temp': float(np.max(temp_matrix)),
            'mean_temp': float(np.mean(temp_matrix)),
            'rows': self.layout.rows,
            'cols': self.layout.cols,
            'n_cells': self.layout.n_cells,
            'configuration': self.configuration.value,
        }

        return temp_matrix, metadata

    def identify_hotspot_patterns(
        self,
        temperature_data: np.ndarray
    ) -> Dict[str, any]:
        """Identify common hotspot patterns.

        Common patterns:
        - Single cell hotspot (defect, shading)
        - Row pattern (bypass diode issue)
        - Column pattern (interconnect issue)
        - Edge pattern (frame issues)
        - Cluster pattern (localized damage)
        """
        if temperature_data.ndim == 1:
            temp_matrix = temperature_data.reshape(
                self.layout.rows, self.layout.cols
            )
        else:
            temp_matrix = temperature_data

        mean_temp = np.mean(temp_matrix)
        threshold = mean_temp + self.HOT_THRESHOLD

        # Create binary hotspot mask
        hotspot_mask = temp_matrix > threshold

        patterns = {
            'single_cell': False,
            'row_pattern': False,
            'column_pattern': False,
            'edge_pattern': False,
            'cluster_pattern': False,
            'pattern_details': []
        }

        if np.sum(hotspot_mask) == 0:
            return patterns

        # Check for row patterns
        row_hotspots = np.sum(hotspot_mask, axis=1)
        if np.any(row_hotspots >= self.layout.cols * 0.5):
            patterns['row_pattern'] = True
            affected_rows = np.where(row_hotspots >= self.layout.cols * 0.5)[0]
            patterns['pattern_details'].append(
                f"Row pattern detected in rows: {affected_rows.tolist()}"
            )

        # Check for column patterns
        col_hotspots = np.sum(hotspot_mask, axis=0)
        if np.any(col_hotspots >= self.layout.rows * 0.5):
            patterns['column_pattern'] = True
            affected_cols = np.where(col_hotspots >= self.layout.rows * 0.5)[0]
            patterns['pattern_details'].append(
                f"Column pattern detected in columns: {affected_cols.tolist()}"
            )

        # Check for edge pattern
        edge_mask = np.zeros_like(hotspot_mask)
        edge_mask[0, :] = hotspot_mask[0, :]  # Top
        edge_mask[-1, :] = hotspot_mask[-1, :]  # Bottom
        edge_mask[:, 0] = hotspot_mask[:, 0]  # Left
        edge_mask[:, -1] = hotspot_mask[:, -1]  # Right

        edge_hotspot_ratio = np.sum(edge_mask) / np.sum(hotspot_mask) if np.sum(hotspot_mask) > 0 else 0
        if edge_hotspot_ratio > 0.6:
            patterns['edge_pattern'] = True
            patterns['pattern_details'].append("Edge pattern detected")

        # Check for clusters using local averaging
        if np.sum(hotspot_mask) > 1:
            smoothed = uniform_filter(hotspot_mask.astype(float), size=3)
            if np.max(smoothed) > 0.5:
                patterns['cluster_pattern'] = True
                patterns['pattern_details'].append("Cluster pattern detected")

        # Single cell
        if np.sum(hotspot_mask) == 1:
            patterns['single_cell'] = True
            row, col = np.where(hotspot_mask)
            patterns['pattern_details'].append(
                f"Single hotspot at cell ({row[0]}, {col[0]})"
            )

        return patterns

    def get_summary_report(self) -> str:
        """Generate text summary of analysis results."""
        if self._results is None:
            return "No analysis has been performed yet."

        r = self._results

        report = f"""
Hotspot Detection Analysis Report
=================================

Configuration: {self.configuration.value}
Layout: {self.layout.rows} x {self.layout.cols} ({self.layout.n_cells} cells)
Number of Strings: {self.layout.n_strings}

Temperature Statistics:
-----------------------
Mean Temperature: {r.mean_temperature:.1f}°C
Std Deviation: {r.std_temperature:.2f}°C
Max ΔT from Mean: {r.max_delta_t:.1f}°C

Cell Classification:
-------------------
Hotspot Cells: {r.hotspot_count}
Weak Cells: {r.weak_cell_count}
Normal Cells: {len(r.normal_cells)}

Uniformity Index: {r.uniformity_index:.1f}%
Overall Status: {r.overall_status.upper()}

String Analysis:
---------------
"""
        for s in r.string_analysis:
            report += f"""
String {s.string_id + 1}:
  Mean Temp: {s.mean_temp:.1f}°C
  Std Dev: {s.std_temp:.2f}°C
  Hotspots: {s.n_hotspots}
  Weak Cells: {s.n_weak_cells}
  Mismatch: {s.mismatch_severity}
"""

        if r.hotspot_cells:
            report += "\nHotspot Details:\n"
            for h in r.hotspot_cells[:5]:  # Top 5
                report += f"  Cell ({h.row}, {h.col}): {h.temperature:.1f}°C (ΔT: {h.delta_t:+.1f}°C)\n"

        return report


def detect_hotspots(
    temperature_data: np.ndarray,
    configuration: CellConfiguration = CellConfiguration.CELLS_72,
    threshold: float = 10.0
) -> HotspotAnalysisResult:
    """Convenience function for quick hotspot detection.

    Args:
        temperature_data: Array of cell temperatures
        configuration: Cell configuration
        threshold: Hot threshold in °C above mean

    Returns:
        HotspotAnalysisResult
    """
    detector = HotspotDetector(
        configuration=configuration,
        hot_threshold=threshold
    )
    return detector.analyze(temperature_data)


def generate_cell_heatmap(
    temperature_data: np.ndarray,
    configuration: CellConfiguration = CellConfiguration.CELLS_72,
    show_strings: bool = True
) -> Tuple[np.ndarray, Dict]:
    """Generate heatmap data for visualization.

    Args:
        temperature_data: Cell temperature data
        configuration: Cell configuration
        show_strings: Whether to include string boundaries

    Returns:
        Tuple of (heatmap matrix, metadata)
    """
    detector = HotspotDetector(configuration=configuration)
    matrix, metadata = detector.generate_heatmap_data(temperature_data)

    if show_strings:
        metadata['string_boundaries'] = _get_string_boundaries(
            detector.layout
        )

    return matrix, metadata


def _get_string_boundaries(layout: CellLayout) -> List[Dict]:
    """Get string boundary coordinates for visualization."""
    boundaries = []
    rows_per_string = layout.rows // layout.n_strings

    if layout.is_half_cut:
        # Vertical boundary in middle
        boundaries.append({
            'type': 'vertical',
            'position': layout.cols // 2,
            'start': 0,
            'end': layout.rows
        })
        # Horizontal boundaries for each half
        for i in range(1, layout.n_strings // 2):
            boundaries.append({
                'type': 'horizontal',
                'position': i * rows_per_string * 2,
                'start': 0,
                'end': layout.cols
            })
    else:
        # Horizontal string boundaries
        for i in range(1, layout.n_strings):
            boundaries.append({
                'type': 'horizontal',
                'position': i * rows_per_string,
                'start': 0,
                'end': layout.cols
            })

    return boundaries


def analyze_string_currents(
    temperature_data: np.ndarray,
    irradiance: float,
    configuration: CellConfiguration = CellConfiguration.CELLS_72,
    isc_stc: float = 10.0
) -> Dict[str, any]:
    """Estimate string currents from temperature distribution.

    Higher temperature cells typically have higher current flow,
    indicating potential mismatch conditions.

    Args:
        temperature_data: Cell temperature data
        irradiance: Measured irradiance (W/m²)
        configuration: Cell configuration
        isc_stc: Short-circuit current at STC (A)

    Returns:
        Dictionary with string current estimates
    """
    detector = HotspotDetector(configuration=configuration)
    result = detector.analyze(temperature_data, irradiance=irradiance)

    # Scale current estimates based on STC value
    irradiance_factor = irradiance / 1000.0

    string_currents = {}
    for s in result.string_analysis:
        # Estimate based on temperature deviation
        temp_factor = 1 - (s.mean_temp - result.mean_temperature) * 0.004
        estimated_current = isc_stc * irradiance_factor * temp_factor

        string_currents[f"string_{s.string_id + 1}"] = {
            'estimated_current': estimated_current,
            'mean_temperature': s.mean_temp,
            'mismatch_severity': s.mismatch_severity,
            'deviation_from_mean': s.mean_temp - result.mean_temperature
        }

    # Calculate mismatch loss
    currents = [v['estimated_current'] for v in string_currents.values()]
    min_current = min(currents)
    max_current = max(currents)
    mismatch_loss = (1 - min_current / max_current) * 100 if max_current > 0 else 0

    string_currents['mismatch_loss_percent'] = mismatch_loss
    string_currents['limiting_string'] = f"string_{currents.index(min_current) + 1}"

    return string_currents
