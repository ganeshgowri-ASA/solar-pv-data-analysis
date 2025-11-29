"""Equipment and Flasher/Simulator Registry.

Supports multiple solar simulator manufacturers and models.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class FlasherEquipment:
    """Solar simulator/flasher equipment details."""
    equipment_id: str
    manufacturer: str
    model: str
    serial_number: str
    qr_code: str = ""
    
    # Calibration information
    calibration_date: datetime = None
    calibration_interval_days: int = 365
    next_calibration_due: datetime = None
    
    # Maintenance
    last_maintenance_date: datetime = None
    maintenance_interval_days: int = 180
    
    # Status
    equipment_status: str = "active"  # active, maintenance, calibration_due, retired
    
    # Technical specifications
    irradiance_range: tuple = (0, 1400)  # W/m²
    temporal_stability: float = 0.02  # ±2%
    spatial_uniformity: float = 0.02  # ±2%
    spectral_match_class: str = "A"  # A, B, C per IEC 60904-9
    
    # Data file format
    file_format: str = "csv"
    delimiter: str = ","
    header_rows: int = 1
    
    def __post_init__(self):
        if self.calibration_date and not self.next_calibration_due:
            self.next_calibration_due = self.calibration_date + timedelta(days=self.calibration_interval_days)
    
    def is_calibration_due(self) -> bool:
        if not self.next_calibration_due:
            return True
        return datetime.now() >= self.next_calibration_due
    
    def is_maintenance_due(self) -> bool:
        if not self.last_maintenance_date:
            return True
        next_maintenance = self.last_maintenance_date + timedelta(days=self.maintenance_interval_days)
        return datetime.now() >= next_maintenance

# ============================================================================
# EQUIPMENT DATABASE
# ============================================================================

EQUIPMENT_DATABASE: Dict[str, FlasherEquipment] = {
    "PASAN": FlasherEquipment(
        equipment_id="PASAN-001",
        manufacturer="PASAN",
        model="HighLIGHT Flasher",
        serial_number="HL-2024-001",
        irradiance_range=(0, 1400),
        temporal_stability=0.01,
        spatial_uniformity=0.02,
        spectral_match_class="A",
        file_format="csv",
        delimiter=",",
    ),
    
    "SPIRE": FlasherEquipment(
        equipment_id="SPIRE-001",
        manufacturer="Spire Corporation",
        model="SPI-SUN 5600",
        serial_number="SP-2024-001",
        irradiance_range=(0, 1200),
        temporal_stability=0.015,
        spatial_uniformity=0.02,
        spectral_match_class="A",
        file_format="txt",
        delimiter="\t",
    ),
    
    "HALM": FlasherEquipment(
        equipment_id="HALM-001",
        manufacturer="Halm Elektronik",
        model="FlashSim",
        serial_number="HE-2024-001",
        irradiance_range=(0, 1300),
        temporal_stability=0.02,
        spatial_uniformity=0.02,
        spectral_match_class="A",
        file_format="csv",
        delimiter=";",
    ),
    
    "MBJ": FlasherEquipment(
        equipment_id="MBJ-001",
        manufacturer="Meyer Burger",
        model="WAVELABS SINUS 3000 ADV",
        serial_number="MB-2024-001",
        irradiance_range=(0, 1400),
        temporal_stability=0.01,
        spatial_uniformity=0.015,
        spectral_match_class="A",
        file_format="xlsx",
        delimiter=",",
    ),
    
    "GSOLAR": FlasherEquipment(
        equipment_id="GSOLAR-001",
        manufacturer="G-Solar",
        model="QuickSun",
        serial_number="GS-2024-001",
        irradiance_range=(0, 1200),
        temporal_stability=0.02,
        spatial_uniformity=0.02,
        spectral_match_class="B",
        file_format="csv",
        delimiter=",",
    ),
    
    "ENDEAS": FlasherEquipment(
        equipment_id="ENDEAS-001",
        manufacturer="Endeas Oy",
        model="Solar Simulator",
        serial_number="EN-2024-001",
        irradiance_range=(0, 1200),
        temporal_stability=0.02,
        spatial_uniformity=0.025,
        spectral_match_class="B",
        file_format="txt",
        delimiter="\t",
    ),
    
    "AVALON": FlasherEquipment(
        equipment_id="AVALON-001",
        manufacturer="Avalon Test Systems",
        model="Flasher IV Tester",
        serial_number="AV-2024-001",
        irradiance_range=(0, 1300),
        temporal_stability=0.015,
        spatial_uniformity=0.02,
        spectral_match_class="A",
        file_format="csv",
        delimiter=",",
    ),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_equipment(manufacturer: str) -> FlasherEquipment:
    """Retrieve equipment details by manufacturer."""
    return EQUIPMENT_DATABASE.get(manufacturer.upper())

def list_all_equipment() -> List[str]:
    """List all registered equipment."""
    return list(EQUIPMENT_DATABASE.keys())

def get_calibration_due_equipment() -> List[FlasherEquipment]:
    """Get list of equipment with calibration due."""
    return [eq for eq in EQUIPMENT_DATABASE.values() if eq.is_calibration_due()]

def get_maintenance_due_equipment() -> List[FlasherEquipment]:
    """Get list of equipment with maintenance due."""
    return [eq for eq in EQUIPMENT_DATABASE.values() if eq.is_maintenance_due()]
