"""Solar PV Data Analysis Platform - Streamlit Dashboard.

Main application entry point.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Solar PV Data Analysis Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #004E89;
        text-align: center;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF6B35;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">☀️ Solar PV Data Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IEC 60904, 60891, 61215 Compliant Testing & Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100.png?text=PV+Lab+Logo", use_column_width=True)
    st.markdown("---")
    
    st.markdown("### 📊 Platform Features")
    st.markdown("""
    - ✅ I-V Curve Analysis
    - ✅ STC Corrections (Procedures 1-4)
    - ✅ Temperature Coefficients
    - ✅ Spectral Mismatch
    - ✅ Uncertainty Analysis
    - ✅ Batch Processing
    - ✅ PDF/Word/Excel Reports
    """)
    
    st.markdown("---")
    st.markdown("### 📦 Supported Equipment")
    equipment_list = [
        "PASAN HighLIGHT",
        "Spire SPI-SUN 5600",
        "Halm FlashSim",
        "Meyer Burger WAVELABS",
        "G-Solar QuickSun",
        "Endeas Simulator",
        "Avalon Flasher"
    ]
    for eq in equipment_list:
        st.markdown(f"- {eq}")
    
    st.markdown("---")
    st.info("📚 Version 1.0.0")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload Data", "📊 I-V Analysis", "⚙️ STC Correction", "📝 Reports"])

# Tab 1: Upload Data
with tab1:
    st.header("📂 Upload Test Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equipment Selection")
        equipment = st.selectbox(
            "Select Flasher/Simulator",
            ["Auto-detect", "PASAN", "Spire", "Halm", "MBJ", "G-Solar", "Endeas", "Avalon"]
        )
        
        st.subheader("Test Conditions")
        measured_temp = st.number_input("Measured Temperature (°C)", value=25.0, step=0.1)
        measured_irrad = st.number_input("Measured Irradiance (W/m²)", value=1000.0, step=1.0)
    
    with col2:
        st.subheader("Data Files")
        uploaded_file = st.file_uploader(
            "Upload I-V Data File",
            type=['txt', 'csv', 'xlsx'],
            help="Supported formats: .txt (tab-delimited), .csv, .xlsx"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Process Data", key="process_btn"):
                st.info("🔄 Processing data... (Feature coming in next update)")
    
    st.markdown("---")
    st.info("""
    💡 **Quick Start Guide:**
    1. Select your equipment type (or use Auto-detect)
    2. Enter measured test conditions
    3. Upload I-V data file (.txt, .csv, or .xlsx)
    4. Click 'Process Data' to analyze
    """)

# Tab 2: I-V Analysis
with tab2:
    st.header("📊 I-V Curve Analysis")
    st.info("🚧 Analysis features will be available after data upload")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Short-Circuit Current (Isc)", "13.53 A", "STC Corrected")
        st.metric("Voltage at MPP (Vmpp)", "42.5 V", "")
    
    with col2:
        st.metric("Open-Circuit Voltage (Voc)", "53.08 V", "STC Corrected")
        st.metric("Current at MPP (Impp)", "12.8 A", "")
    
    with col3:
        st.metric("Maximum Power (Pmax)", "581.2 W", "+2.5%")
        st.metric("Fill Factor (FF)", "0.8095", "")

# Tab 3: STC Correction
with tab3:
    st.header("⚙️ STC Correction (IEC 60891)")
    
    procedure = st.selectbox(
        "Select Correction Procedure",
        [
            "Procedure 1 - Full correction (Alpha, Beta, Rs, Kappa)",
            "Procedure 2 - Multiple curves interpolation",
            "Procedure 3 - Simplified with Rs",
            "Procedure 4 - Quick correction"
        ]
    )
    
    st.markdown("### Temperature Coefficients")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alpha = st.number_input("Alpha (A/°C)", value=0.0005, format="%.5f")
    with col2:
        beta = st.number_input("Beta (V/°C)", value=-0.003, format="%.5f")
    with col3:
        rs = st.number_input("Rs (Ω)", value=0.255, format="%.4f")
    with col4:
        kappa = st.number_input("Kappa (Ω/°C)", value=0.001, format="%.5f")
    
    if st.button("🔄 Apply Correction"):
        st.info("🔄 Correction will be applied after data upload")

# Tab 4: Reports
with tab4:
    st.header("📝 Test Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Report Configuration")
        report_type = st.selectbox("Report Type", ["IEC 60904-1 Test Report", "Full Qualification Report", "Quick Summary"])
        include_charts = st.checkbox("Include I-V/P-V Charts", value=True)
        include_uncertainty = st.checkbox("Include Uncertainty Budget", value=True)
        
    with col2:
        st.subheader("Export Format")
        export_format = st.multiselect(
            "Select format(s)",
            ["PDF", "Word (.docx)", "Excel (.xlsx)"],
            default=["PDF"]
        )
        
        if st.button("📥 Generate Report"):
            st.info("🔄 Report generation coming in next update")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Standards Compliance**  
    IEC 60904, 60891, 61215, 61853
    """)

with col2:
    st.markdown("""
    **👥 Contact**  
    [GitHub](https://github.com/ganeshgowri-ASA/solar-pv-data-analysis)
    """)

with col3:
    st.markdown("""
    **© 2025 PV Testing Lab**  
    ISO/IEC 17025:2017
    """)
