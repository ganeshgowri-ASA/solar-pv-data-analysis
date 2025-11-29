# Solar PV Data Analysis Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-316192.svg)](https://www.postgresql.org/)

Comprehensive Solar PV Module Testing & Analysis Platform with full IEC standards compliance, multi-flasher support, and LIMS integration capabilities.

## 🎯 Overview

A production-ready, enterprise-grade platform for solar photovoltaic module testing and data analysis. Designed for testing laboratories, manufacturers, and research institutions requiring IEC-compliant test procedures, automated data processing, uncertainty quantification, and comprehensive reporting.

### Key Capabilities

- **✅ IEC Standards Compliant**: IEC 60904, 60891, 61215, 61853
- **📊 I-V Curve Analysis**: Complete characterization with STC corrections
- **🌡️ Temperature Coefficients**: Alpha, Beta, Gamma, Rs extraction
- **🌈 Spectral Mismatch Correction**: IEC 60904-7 compliant
- **🎯 Uncertainty Quantification**: GUM-compliant propagation
- **🔥 Hotspot Detection**: Statistical + thermal imaging + physical analysis
- **📊 Energy Rating**: IEC 61853 multi-condition analysis
- **📦 Multi-Flasher Support**: Avalon, PASAN, Endeas, Gsola, Halm, MBJ, Spire
- **📦 LIMS Integration**: Equipment tracking, QR codes, personnel, deviations

---

## 🚀 Features

### 🔬 Test Procedures

#### I-V Curve Correction (IEC 60891)
- **Procedure 1**: Translation using measured temperature coefficients (Alpha, Beta, Rs, Kappa)
- **Procedure 2**: Translation using multiple curves at different conditions
- **Procedure 3**: Simplified correction with series resistance
- **Procedure 4**: Quick correction without series resistance

#### Temperature Coefficient Measurement
- Alpha (Isc temperature coefficient)
- Beta (Voc temperature coefficient)
- Gamma (Pmax temperature coefficient)
- Series Resistance (Rs)
- Curve correction factor (Kappa)

#### Spectral Mismatch Correction (IEC 60904-7)
- Multi-bin spectral analysis (300-1200 nm)
- Reference cell vs. DUT spectral response
- Simulator spectrum characterization
- Mismatch factor calculation

### 📦 Equipment Support

Built-in support for major solar simulator manufacturers:

| Manufacturer | Model | Spectral Class | Status |
|--------------|-------|----------------|--------|
| **PASAN** | HighLIGHT Flasher | A | ✅ Supported |
| **Spire** | SPI-SUN 5600 | A | ✅ Supported |
| **Halm Elektronik** | FlashSim | A | ✅ Supported |
| **Meyer Burger (MBJ)** | WAVELABS SINUS 3000 | A | ✅ Supported |
| **G-Solar** | QuickSun | B | ✅ Supported |
| **Endeas** | Solar Simulator | B | ✅ Supported |
| **Avalon** | Flasher IV Tester | A | ✅ Supported |

*More equipment can be added incrementally via `config/equipment_registry.py`*

### 📊 Quality Control & LIMS

- **Equipment Management**: Calibration tracking, maintenance schedules, QR codes
- **Sample Tracking**: Project-based organization, QR code identification
- **Personnel Management**: Technician/reviewer assignment, accountability
- **Auto-Deviation Detection**: Real-time flagging with severity levels (Warning/Critical)
- **Uncertainty Propagation**: GUM-compliant combined and expanded uncertainties
- **Statement of Conformity**: Automated generation based on pass/fail criteria
- **Control Charts**: Statistical process control for repeatability monitoring

### 📊 Reporting

**Export Formats**:
- 📄 PDF Reports (IEC-compliant templates)
- 📄 Word Documents (.docx)
- 📊 Excel Spreadsheets (.xlsx)

**Report Content**:
- Complete test methodology and standards references
- I-V curves (raw and corrected)
- Performance metrics at STC
- Temperature coefficients
- Uncertainty budget breakdown
- Equipment calibration status
- Environmental conditions
- Personnel signatures
- Statement of conformity

---

## 💻 Architecture

### Repository Structure

```
solar-pv-data-analysis/
├── README.md
├── requirements.txt
├── config/
│   ├── config.py                 # Paths, environment, DB connections
│   ├── iec_standards.py          # IEC 60904, 60891, 61215 parameters
│   └── equipment_registry.py     # Flasher/simulator registry
├── data/
│   ├── raw/                      # Original flasher data
│   ├── processed/                # Cleaned/curated
│   └── computed/                 # Results, plots
├── src/
│   ├── ingestion/                # Data loaders (CSV, XLSX, TXT, SR files)
│   ├── analysis/
│   │   ├── iv_curve.py           # I-V characterization (IEC 60904-1)
│   │   ├── corrections.py        # Procedure 1-4 (IEC 60891)
│   │   ├── spectral_mismatch.py  # IEC 60904-7
│   │   ├── temperature_coeff.py  # Alpha, Beta, Rs, Kappa
│   │   ├── uncertainty.py        # GUM uncertainty propagation
│   │   └── hotspot_detection.py  # Statistical + visual + physical
│   ├── reporting/
│   │   ├── word_export.py
│   │   ├── pdf_export.py
│   │   └── excel_export.py
│   └── database/
│       └── schema.py             # PostgreSQL schema
├── notebooks/                    # Jupyter exploration
├── tests/
└── docs/
    └── IEC_compliance.md
```

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ganeshgowri-ASA/solar-pv-data-analysis.git
cd solar-pv-data-analysis

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure database (Railway PostgreSQL)
cp .env.example .env
# Edit .env with DATABASE_URL

# Run application
streamlit run app.py
```

---

## 📝 Roadmap

### v1.0 (Current)
- [x] Core I-V analysis
- [x] IEC 60891 Procedures 1-4
- [x] Database schema with LIMS
- [x] Multi-flasher support

### v1.1 (Next)
- [ ] Streamlit web interface
- [ ] Interactive dashboards
- [ ] PDF/Word/Excel export
- [ ] Temperature coefficient extraction
- [ ] Energy rating analysis

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and create Pull Request

**Contribution Areas**:
- Additional equipment support
- New analysis algorithms
- Documentation improvements
- Test coverage

---

## 📜 Standards

- IEC 60904-1: I-V characteristics measurement
- IEC 60891: Temperature/irradiance corrections
- IEC 61215: Design qualification
- IEC 61853: Energy rating
- GUM: Uncertainty in measurement

---

## 📧 Contact

**Maintainer**: Ganesh Gowri  
**GitHub**: [@ganeshgowri-ASA](https://github.com/ganeshgowri-ASA)  
**Repository**: [solar-pv-data-analysis](https://github.com/ganeshgowri-ASA/solar-pv-data-analysis)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**⭐ Star this repository if you find it useful!**
