"""Equipment and Flasher/Simulator Registry.

Comprehensive registry of solar simulator manufacturers and models
with support for calibration tracking, file format detection, and
equipment-specific parsing configurations.

Supported Manufacturers:
- PASAN (Switzerland)
- Spire Corporation (USA)
- Halm Elektronik (Germany)
- MBJ Solutions (Germany)
- Wavelabs Solar Metrology Systems (Germany)
- Endeas Oy (Finland)
- Avalon Test Systems (USA)
- Quicksun (Multiple)
- G-Solar (China)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re


class EquipmentStatus(Enum):
    """Equipment operational status."""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    CALIBRATION_DUE = "calibration_due"
    OUT_OF_SERVICE = "out_of_service"
    RETIRED = "retired"


class FileFormatType(Enum):
    """Supported file format types."""
    CSV = "csv"
    TXT = "txt"
    XLSX = "xlsx"
    XLS = "xls"
    DAT = "dat"
    XML = "xml"
    JSON = "json"


@dataclass
class DataFormat:
    """Data format specification for equipment output files."""
    file_extension: FileFormatType
    delimiter: str = ","
    header_rows: int = 1
    skip_footer: int = 0
    encoding: str = "utf-8"
    decimal_separator: str = "."
    thousands_separator: str = ""

    # Column mapping
    voltage_column: Optional[str] = None
    current_column: Optional[str] = None
    power_column: Optional[str] = None
    irradiance_column: Optional[str] = None
    temperature_column: Optional[str] = None

    # Data location (for Excel files)
    sheet_name: Optional[str] = None
    data_start_row: Optional[int] = None
    data_start_col: Optional[int] = None

    # Special parsing rules
    parse_metadata_from_header: bool = True
    metadata_rows: int = 10


@dataclass
class CalibrationInfo:
    """Calibration information for equipment."""
    calibration_date: Optional[datetime] = None
    calibration_interval_days: int = 365
    calibration_lab: str = ""
    certificate_number: str = ""
    uncertainty_k2: float = 0.0  # Expanded uncertainty at k=2
    reference_cell_id: str = ""
    traceability: str = ""

    @property
    def next_calibration_due(self) -> Optional[datetime]:
        """Calculate next calibration due date."""
        if self.calibration_date:
            return self.calibration_date + timedelta(days=self.calibration_interval_days)
        return None

    def is_calibration_due(self) -> bool:
        """Check if calibration is due."""
        if not self.next_calibration_due:
            return True
        return datetime.now() >= self.next_calibration_due

    def days_until_calibration(self) -> Optional[int]:
        """Return days until calibration is due (negative if overdue)."""
        if not self.next_calibration_due:
            return None
        delta = self.next_calibration_due - datetime.now()
        return delta.days


@dataclass
class TechnicalSpecs:
    """Technical specifications of solar simulator."""
    # Irradiance specifications
    irradiance_range_min: float = 0.0  # W/m²
    irradiance_range_max: float = 1400.0  # W/m²
    irradiance_uniformity: float = 2.0  # ±%

    # Temporal specifications
    flash_duration_ms: Optional[float] = None  # For flash simulators
    temporal_stability_sti: float = 0.5  # % (short-term)
    temporal_stability_lti: float = 2.0  # % (long-term)

    # Spectral specifications
    spectral_match_class: str = "A"  # A+, A, B, C
    spectrum_type: str = "Xenon"  # Xenon, LED, Halogen, Multi-source

    # Overall classification (per IEC 60904-9)
    classification: str = "AAA"  # Spectral, Uniformity, Temporal

    # Test area
    test_area_width_mm: float = 2200.0
    test_area_height_mm: float = 1200.0

    # Measurement capabilities
    max_voltage: float = 100.0  # V
    max_current: float = 20.0  # A
    voltage_resolution: float = 0.001  # V
    current_resolution: float = 0.0001  # A

    # IV curve measurement
    iv_points_per_curve: int = 200
    measurement_speed_ms: float = 10.0  # ms per point


@dataclass
class FlasherEquipment:
    """Complete solar simulator/flasher equipment definition."""
    equipment_id: str
    manufacturer: str
    model: str
    serial_number: str
    qr_code: str = ""

    # Technical specifications
    specs: TechnicalSpecs = field(default_factory=TechnicalSpecs)

    # Data format
    data_format: DataFormat = field(default_factory=DataFormat)

    # Calibration information
    calibration: CalibrationInfo = field(default_factory=CalibrationInfo)

    # Maintenance tracking
    last_maintenance_date: Optional[datetime] = None
    maintenance_interval_days: int = 180
    maintenance_notes: str = ""

    # Status
    status: EquipmentStatus = EquipmentStatus.ACTIVE

    # File pattern matching (regex patterns for auto-detection)
    filename_patterns: List[str] = field(default_factory=list)
    content_patterns: List[str] = field(default_factory=list)

    # Additional metadata
    location: str = ""
    responsible_person: str = ""
    purchase_date: Optional[datetime] = None
    notes: str = ""

    def is_calibration_due(self) -> bool:
        """Check if calibration is due."""
        return self.calibration.is_calibration_due()

    def is_maintenance_due(self) -> bool:
        """Check if maintenance is due."""
        if not self.last_maintenance_date:
            return True
        next_maintenance = self.last_maintenance_date + timedelta(
            days=self.maintenance_interval_days
        )
        return datetime.now() >= next_maintenance

    def get_status_string(self) -> str:
        """Get comprehensive status string."""
        status_parts = [f"Status: {self.status.value}"]

        if self.is_calibration_due():
            status_parts.append("CALIBRATION DUE")
        elif self.calibration.days_until_calibration():
            days = self.calibration.days_until_calibration()
            if days and days < 30:
                status_parts.append(f"Calibration in {days} days")

        if self.is_maintenance_due():
            status_parts.append("MAINTENANCE DUE")

        return " | ".join(status_parts)


# ============================================================================
# EQUIPMENT DATABASE - Complete registry of all supported equipment
# ============================================================================

EQUIPMENT_DATABASE: Dict[str, FlasherEquipment] = {
    # -------------------------------------------------------------------------
    # PASAN (Switzerland) - High-precision flash testers
    # -------------------------------------------------------------------------
    "PASAN": FlasherEquipment(
        equipment_id="PASAN-001",
        manufacturer="PASAN SA",
        model="HighLIGHT Flasher",
        serial_number="HL-2024-001",
        qr_code="PASAN-HL-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1400,
            irradiance_uniformity=1.0,
            flash_duration_ms=10.0,
            temporal_stability_sti=0.2,
            temporal_stability_lti=1.0,
            spectral_match_class="A+",
            spectrum_type="Xenon",
            classification="A+A+A+",
            test_area_width_mm=2200,
            test_area_height_mm=1200,
            iv_points_per_curve=500,
            measurement_speed_ms=2.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=",",
            header_rows=15,
            voltage_column="Voltage (V)",
            current_column="Current (A)",
            power_column="Power (W)",
            irradiance_column="Irradiance (W/m2)",
            temperature_column="Temperature (C)",
        ),
        filename_patterns=[r"pasan", r"highlight", r"hl[\-_]?\d+"],
        content_patterns=[r"PASAN", r"HighLIGHT", r"Flasher Report"],
    ),

    # -------------------------------------------------------------------------
    # Spire Corporation (USA) - Industrial flash testers
    # -------------------------------------------------------------------------
    "SPIRE": FlasherEquipment(
        equipment_id="SPIRE-001",
        manufacturer="Spire Corporation",
        model="SPI-SUN 5600SLP",
        serial_number="SP-2024-001",
        qr_code="SPIRE-5600-001",
        specs=TechnicalSpecs(
            irradiance_range_min=50,
            irradiance_range_max=1200,
            irradiance_uniformity=2.0,
            flash_duration_ms=20.0,
            temporal_stability_sti=0.5,
            temporal_stability_lti=2.0,
            spectral_match_class="A",
            spectrum_type="Xenon",
            classification="AAA",
            test_area_width_mm=2000,
            test_area_height_mm=1100,
            iv_points_per_curve=400,
            measurement_speed_ms=5.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.TXT,
            delimiter="\t",
            header_rows=20,
            voltage_column="V",
            current_column="I",
            encoding="utf-8",
        ),
        filename_patterns=[r"spire", r"spi[\-_]?sun", r"sp[\-_]?\d+"],
        content_patterns=[r"Spire Corporation", r"SPI-SUN", r"SPISUN"],
    ),

    # -------------------------------------------------------------------------
    # Halm Elektronik (Germany) - cetisPV series
    # -------------------------------------------------------------------------
    "HALM": FlasherEquipment(
        equipment_id="HALM-001",
        manufacturer="Halm Elektronik GmbH",
        model="cetisPV-CTL4",
        serial_number="HE-2024-001",
        qr_code="HALM-CTL4-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1300,
            irradiance_uniformity=1.5,
            flash_duration_ms=15.0,
            temporal_stability_sti=0.3,
            temporal_stability_lti=1.5,
            spectral_match_class="A",
            spectrum_type="Xenon",
            classification="AAA",
            test_area_width_mm=2100,
            test_area_height_mm=1100,
            iv_points_per_curve=300,
            measurement_speed_ms=3.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=";",
            header_rows=10,
            decimal_separator=",",
            voltage_column="Voltage",
            current_column="Current",
        ),
        filename_patterns=[r"halm", r"cetis", r"ctl[\-_]?\d+", r"he[\-_]?\d+"],
        content_patterns=[r"Halm", r"cetisPV", r"Flasher Test"],
    ),

    # -------------------------------------------------------------------------
    # MBJ Solutions (Germany) - LED-based simulators
    # -------------------------------------------------------------------------
    "MBJ": FlasherEquipment(
        equipment_id="MBJ-001",
        manufacturer="MBJ Solutions GmbH",
        model="LED Sun Simulator",
        serial_number="MBJ-2024-001",
        qr_code="MBJ-LED-001",
        specs=TechnicalSpecs(
            irradiance_range_min=50,
            irradiance_range_max=1400,
            irradiance_uniformity=1.0,
            flash_duration_ms=None,  # Continuous LED
            temporal_stability_sti=0.1,
            temporal_stability_lti=0.5,
            spectral_match_class="A+",
            spectrum_type="LED",
            classification="A+A+A+",
            test_area_width_mm=2200,
            test_area_height_mm=1300,
            iv_points_per_curve=500,
            measurement_speed_ms=2.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.XLSX,
            sheet_name="IV Data",
            voltage_column="U [V]",
            current_column="I [A]",
            data_start_row=5,
        ),
        filename_patterns=[r"mbj", r"led[\-_]?sun"],
        content_patterns=[r"MBJ Solutions", r"LED Sun Simulator"],
    ),

    # -------------------------------------------------------------------------
    # Wavelabs Solar Metrology Systems (Germany) - SINUS series
    # -------------------------------------------------------------------------
    "WAVELABS": FlasherEquipment(
        equipment_id="WAVELABS-001",
        manufacturer="Wavelabs Solar Metrology Systems GmbH",
        model="SINUS-3600 ADV",
        serial_number="WL-2024-001",
        qr_code="WAVELABS-3600-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1400,
            irradiance_uniformity=0.8,
            flash_duration_ms=None,  # Continuous LED
            temporal_stability_sti=0.1,
            temporal_stability_lti=0.3,
            spectral_match_class="A+",
            spectrum_type="LED",
            classification="A+A+A+",
            test_area_width_mm=2400,
            test_area_height_mm=1400,
            iv_points_per_curve=1000,
            measurement_speed_ms=1.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=",",
            header_rows=12,
            voltage_column="Voltage [V]",
            current_column="Current [A]",
            power_column="Power [W]",
        ),
        filename_patterns=[r"wavelabs", r"sinus[\-_]?\d+", r"wl[\-_]?\d+"],
        content_patterns=[r"Wavelabs", r"SINUS", r"Solar Metrology"],
    ),

    # -------------------------------------------------------------------------
    # Endeas Oy (Finland) - QuickSun simulators
    # -------------------------------------------------------------------------
    "ENDEAS": FlasherEquipment(
        equipment_id="ENDEAS-001",
        manufacturer="Endeas Oy",
        model="QuickSun 540",
        serial_number="EN-2024-001",
        qr_code="ENDEAS-QS540-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1200,
            irradiance_uniformity=2.0,
            flash_duration_ms=25.0,
            temporal_stability_sti=0.5,
            temporal_stability_lti=2.0,
            spectral_match_class="A",
            spectrum_type="Xenon",
            classification="AAA",
            test_area_width_mm=1800,
            test_area_height_mm=1000,
            iv_points_per_curve=200,
            measurement_speed_ms=5.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.TXT,
            delimiter="\t",
            header_rows=8,
            voltage_column="V",
            current_column="I",
        ),
        filename_patterns=[r"endeas", r"quicksun[\-_]?540", r"qs[\-_]?\d+", r"en[\-_]?\d+"],
        content_patterns=[r"Endeas", r"QuickSun"],
    ),

    # -------------------------------------------------------------------------
    # Quicksun (standalone brand entry for broader compatibility)
    # -------------------------------------------------------------------------
    "QUICKSUN": FlasherEquipment(
        equipment_id="QUICKSUN-001",
        manufacturer="Endeas Oy / Various",
        model="QuickSun Series",
        serial_number="QS-2024-001",
        qr_code="QUICKSUN-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1200,
            irradiance_uniformity=2.5,
            flash_duration_ms=20.0,
            temporal_stability_sti=0.5,
            temporal_stability_lti=2.5,
            spectral_match_class="B",
            spectrum_type="Xenon",
            classification="ABB",
            test_area_width_mm=1600,
            test_area_height_mm=1000,
            iv_points_per_curve=150,
            measurement_speed_ms=8.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=",",
            header_rows=5,
            voltage_column="Voltage",
            current_column="Current",
        ),
        filename_patterns=[r"quicksun", r"qs[\-_]?[a-z]*\d*"],
        content_patterns=[r"QuickSun", r"QS\d+"],
    ),

    # -------------------------------------------------------------------------
    # Avalon Test Systems (USA) - Precision testers
    # -------------------------------------------------------------------------
    "AVALON": FlasherEquipment(
        equipment_id="AVALON-001",
        manufacturer="Avalon Test Systems",
        model="FlashMaster 2000",
        serial_number="AV-2024-001",
        qr_code="AVALON-FM2000-001",
        specs=TechnicalSpecs(
            irradiance_range_min=50,
            irradiance_range_max=1300,
            irradiance_uniformity=1.5,
            flash_duration_ms=15.0,
            temporal_stability_sti=0.3,
            temporal_stability_lti=1.5,
            spectral_match_class="A",
            spectrum_type="Xenon",
            classification="AAA",
            test_area_width_mm=2000,
            test_area_height_mm=1200,
            iv_points_per_curve=350,
            measurement_speed_ms=4.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=",",
            header_rows=10,
            voltage_column="Voltage (V)",
            current_column="Current (A)",
        ),
        filename_patterns=[r"avalon", r"flashmaster", r"fm[\-_]?\d+", r"av[\-_]?\d+"],
        content_patterns=[r"Avalon", r"FlashMaster", r"Test Systems"],
    ),

    # -------------------------------------------------------------------------
    # G-Solar (China) - Production-grade testers
    # -------------------------------------------------------------------------
    "GSOLAR": FlasherEquipment(
        equipment_id="GSOLAR-001",
        manufacturer="G-Solar Technology Co., Ltd.",
        model="GS-SST-1000",
        serial_number="GS-2024-001",
        qr_code="GSOLAR-SST1000-001",
        specs=TechnicalSpecs(
            irradiance_range_min=100,
            irradiance_range_max=1200,
            irradiance_uniformity=2.0,
            flash_duration_ms=20.0,
            temporal_stability_sti=0.5,
            temporal_stability_lti=2.0,
            spectral_match_class="A",
            spectrum_type="Xenon",
            classification="AAA",
            test_area_width_mm=2100,
            test_area_height_mm=1200,
            iv_points_per_curve=250,
            measurement_speed_ms=5.0,
        ),
        data_format=DataFormat(
            file_extension=FileFormatType.CSV,
            delimiter=",",
            header_rows=8,
            voltage_column="V",
            current_column="I",
        ),
        filename_patterns=[r"g[\-_]?solar", r"gs[\-_]?sst", r"gs[\-_]?\d+"],
        content_patterns=[r"G-Solar", r"GSolar", r"GS-SST"],
    ),
}


# ============================================================================
# MANUFACTURER ALIASES - Map various names to canonical keys
# ============================================================================

MANUFACTURER_ALIASES: Dict[str, str] = {
    # PASAN
    "pasan": "PASAN",
    "highlight": "PASAN",
    "pasan sa": "PASAN",

    # Spire
    "spire": "SPIRE",
    "spire corporation": "SPIRE",
    "spi-sun": "SPIRE",
    "spisun": "SPIRE",

    # Halm
    "halm": "HALM",
    "halm elektronik": "HALM",
    "cetis": "HALM",
    "cetispv": "HALM",

    # MBJ
    "mbj": "MBJ",
    "mbj solutions": "MBJ",
    "led sun": "MBJ",

    # Wavelabs
    "wavelabs": "WAVELABS",
    "sinus": "WAVELABS",
    "wavelabs solar": "WAVELABS",

    # Endeas
    "endeas": "ENDEAS",
    "endeas oy": "ENDEAS",

    # Quicksun
    "quicksun": "QUICKSUN",
    "quick sun": "QUICKSUN",
    "qs": "QUICKSUN",

    # Avalon
    "avalon": "AVALON",
    "avalon test": "AVALON",
    "flashmaster": "AVALON",

    # G-Solar
    "g-solar": "GSOLAR",
    "gsolar": "GSOLAR",
    "g solar": "GSOLAR",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_equipment(identifier: str) -> Optional[FlasherEquipment]:
    """Retrieve equipment by manufacturer name or alias.

    Args:
        identifier: Manufacturer name, alias, or equipment ID

    Returns:
        FlasherEquipment object or None if not found
    """
    # Normalize input
    identifier_lower = identifier.lower().strip()

    # Check aliases first
    canonical_key = MANUFACTURER_ALIASES.get(identifier_lower)
    if canonical_key:
        return EQUIPMENT_DATABASE.get(canonical_key)

    # Check direct match (case-insensitive)
    for key in EQUIPMENT_DATABASE:
        if key.lower() == identifier_lower:
            return EQUIPMENT_DATABASE[key]

    # Check equipment_id
    for equipment in EQUIPMENT_DATABASE.values():
        if equipment.equipment_id.lower() == identifier_lower:
            return equipment

    return None


def detect_equipment_from_filename(filename: str) -> Optional[str]:
    """Detect equipment type from filename.

    Args:
        filename: Name of the data file

    Returns:
        Canonical equipment key or None
    """
    filename_lower = filename.lower()

    for key, equipment in EQUIPMENT_DATABASE.items():
        for pattern in equipment.filename_patterns:
            if re.search(pattern, filename_lower):
                return key

    return None


def detect_equipment_from_content(content: str) -> Optional[str]:
    """Detect equipment type from file content.

    Args:
        content: First portion of file content (e.g., first 1KB)

    Returns:
        Canonical equipment key or None
    """
    for key, equipment in EQUIPMENT_DATABASE.items():
        for pattern in equipment.content_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return key

    return None


def list_all_equipment() -> List[str]:
    """List all registered equipment keys.

    Returns:
        List of canonical equipment keys
    """
    return list(EQUIPMENT_DATABASE.keys())


def list_all_manufacturers() -> List[str]:
    """List all unique manufacturers.

    Returns:
        List of manufacturer names
    """
    manufacturers = set()
    for equipment in EQUIPMENT_DATABASE.values():
        manufacturers.add(equipment.manufacturer)
    return sorted(manufacturers)


def get_calibration_due_equipment() -> List[FlasherEquipment]:
    """Get list of equipment with calibration due.

    Returns:
        List of equipment needing calibration
    """
    return [
        eq for eq in EQUIPMENT_DATABASE.values()
        if eq.is_calibration_due()
    ]


def get_maintenance_due_equipment() -> List[FlasherEquipment]:
    """Get list of equipment with maintenance due.

    Returns:
        List of equipment needing maintenance
    """
    return [
        eq for eq in EQUIPMENT_DATABASE.values()
        if eq.is_maintenance_due()
    ]


def get_equipment_by_classification(
    min_class: str = "A"
) -> List[FlasherEquipment]:
    """Get equipment meeting minimum classification.

    Args:
        min_class: Minimum classification (A+, A, B, or C)

    Returns:
        List of qualifying equipment
    """
    class_order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    min_order = class_order.get(min_class, 1)

    result = []
    for equipment in EQUIPMENT_DATABASE.values():
        # Check all three classification letters
        classification = equipment.specs.classification
        if len(classification) >= 3:
            # Handle A+ which takes 2 characters
            parts = []
            i = 0
            while i < len(classification):
                if i + 1 < len(classification) and classification[i + 1] == '+':
                    parts.append(classification[i:i + 2])
                    i += 2
                else:
                    parts.append(classification[i])
                    i += 1

            # Check if all parts meet minimum
            if all(class_order.get(p, 3) <= min_order for p in parts[:3]):
                result.append(equipment)

    return result


def get_equipment_summary() -> Dict[str, Any]:
    """Get summary statistics of equipment registry.

    Returns:
        Dictionary with summary statistics
    """
    total = len(EQUIPMENT_DATABASE)
    cal_due = len(get_calibration_due_equipment())
    maint_due = len(get_maintenance_due_equipment())

    by_status = {}
    by_type = {}

    for equipment in EQUIPMENT_DATABASE.values():
        # Count by status
        status = equipment.status.value
        by_status[status] = by_status.get(status, 0) + 1

        # Count by spectrum type
        spec_type = equipment.specs.spectrum_type
        by_type[spec_type] = by_type.get(spec_type, 0) + 1

    return {
        "total_equipment": total,
        "calibration_due": cal_due,
        "maintenance_due": maint_due,
        "by_status": by_status,
        "by_spectrum_type": by_type,
        "manufacturers": list_all_manufacturers(),
    }
