"""Solar PV Data Analysis Platform - Streamlit Dashboard.

Comprehensive I-V curve analysis, STC corrections, and reporting
per IEC 60904, 60891, and 61215 standards.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules
try:
    from src.analysis.iv_curve import IVCurveAnalyzer
    from src.analysis.corrections import (
        CorrectionProcedure1,
        CorrectionProcedure3,
        CorrectionProcedure4
    )
    from src.analysis.uncertainty import create_pmax_uncertainty_budget
    from src.analysis.gates_calculation import power_bin_classification
    from config.equipment_registry import list_all_equipment, get_equipment
    from config.iec_standards import TYPICAL_TEMP_COEFFICIENTS, CORRECTION_PROCEDURES
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)

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
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF6B35;
        color: white;
    }
    .stButton>button:hover {
        background-color: #e55a2b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'iv_data' not in st.session_state:
    st.session_state.iv_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'stc_results' not in st.session_state:
    st.session_state.stc_results = None


def generate_sample_iv_curve():
    """Generate sample I-V curve data for demonstration."""
    isc = 13.5
    voc = 53.0
    n_points = 200
    voltage = np.linspace(0, voc, n_points)

    # Simplified single-diode model
    rs = 0.3
    rsh = 500
    n = 1.3
    vt = 0.026 * 72  # 72 cells
    i0 = 1e-10

    current = np.zeros_like(voltage)
    for i, v in enumerate(voltage):
        i_guess = isc
        for _ in range(20):
            i_new = isc - i0 * (np.exp((v + i_guess * rs) / (n * vt)) - 1) - (v + i_guess * rs) / rsh
            if abs(i_new - i_guess) < 1e-6:
                break
            i_guess = i_new
        current[i] = max(0, i_new)

    return voltage, current


def create_iv_plot(voltage, current, results=None):
    """Create I-V and P-V curves using Streamlit native charts."""
    power = voltage * current

    # Create dataframes for plotting
    iv_df = pd.DataFrame({'Voltage (V)': voltage, 'Current (A)': current})
    pv_df = pd.DataFrame({'Voltage (V)': voltage, 'Power (W)': power})

    return iv_df, pv_df


# Main header
st.markdown('<div class="main-header">☀️ Solar PV Data Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IEC 60904, 60891, 61215 Compliant Testing & Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("#### Equipment Selection")
    if MODULES_LOADED:
        equipment_options = ["Auto-detect"] + list_all_equipment()
    else:
        equipment_options = ["Auto-detect", "PASAN", "SPIRE", "HALM", "MBJ", "WAVELABS", "ENDEAS", "AVALON", "GSOLAR"]

    selected_equipment = st.selectbox("Flasher/Simulator", equipment_options)

    if selected_equipment != "Auto-detect" and MODULES_LOADED:
        eq = get_equipment(selected_equipment)
        if eq:
            st.info(f"**{eq.manufacturer}**\n\nModel: {eq.model}\n\nClass: {eq.specs.classification}")

    st.markdown("---")
    st.markdown("#### Test Conditions")
    measured_temp = st.number_input("Temperature (°C)", value=25.0, step=0.1, format="%.1f")
    measured_irrad = st.number_input("Irradiance (W/m²)", value=1000.0, step=1.0, format="%.0f")

    st.markdown("---")
    st.markdown("#### Module Specifications")
    rated_power = st.number_input("Rated Power (Wp)", value=550.0, step=5.0)
    rated_voc = st.number_input("Rated Voc (V)", value=53.0, step=0.1)
    rated_isc = st.number_input("Rated Isc (A)", value=13.5, step=0.1)

    st.markdown("---")
    st.markdown("### 📚 Standards")
    st.markdown("""
    - IEC 60904-1: I-V Curves
    - IEC 60891: STC Correction
    - IEC 60904-9: Simulator Class
    - IEC 60904-7: Spectral Match
    - IEC 60904-10: Temp Coefficients
    """)

    st.markdown("---")
    st.info("📦 Version 2.0.0")

# Check if modules loaded
if not MODULES_LOADED:
    st.warning(f"Some modules could not be loaded: {IMPORT_ERROR}. Running in demo mode.")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Upload", "📊 I-V Analysis", "⚙️ STC Correction",
    "📈 Uncertainty", "📝 Reports"
])

# Tab 1: Data Upload
with tab1:
    st.header("📂 Upload Test Data")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Upload I-V Data File",
            type=['txt', 'csv', 'xlsx'],
            help="Supported formats: .txt (tab-delimited), .csv, .xlsx"
        )

        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"File size: {file_size:.1f} KB")

        st.markdown("---")
        st.subheader("Or Use Demo Data")
        if st.button("🔬 Load Sample I-V Curve", key="demo_btn"):
            voltage, current = generate_sample_iv_curve()
            st.session_state.iv_data = {'voltage': voltage, 'current': current}
            st.success("✅ Sample data loaded successfully!")
            st.rerun()

    with col2:
        st.subheader("Quick Analysis Preview")

        if st.session_state.iv_data is not None:
            voltage = st.session_state.iv_data['voltage']
            current = st.session_state.iv_data['current']

            if MODULES_LOADED:
                analyzer = IVCurveAnalyzer(voltage, current)
                results = analyzer.extract_parameters()
                st.session_state.analysis_results = results

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Isc", f"{results['isc']:.3f} A")
                    st.metric("Vmpp", f"{results['vmpp']:.2f} V")
                    st.metric("Pmax", f"{results['pmax']:.1f} W")
                with col_b:
                    st.metric("Voc", f"{results['voc']:.2f} V")
                    st.metric("Impp", f"{results['impp']:.3f} A")
                    st.metric("FF", f"{results['ff']:.4f}")

            # Simple I-V plot
            iv_df, pv_df = create_iv_plot(voltage, current)
            st.line_chart(iv_df.set_index('Voltage (V)'))
        else:
            st.info("👆 Upload a file or load demo data to see analysis")

# Tab 2: I-V Analysis
with tab2:
    st.header("📊 I-V Curve Analysis")

    if st.session_state.iv_data is not None and st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        voltage = st.session_state.iv_data['voltage']
        current = st.session_state.iv_data['current']

        # Main metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            delta = f"{((results['isc']/rated_isc)-1)*100:+.1f}%" if rated_isc else None
            st.metric("Isc", f"{results['isc']:.3f} A", delta=delta)
        with col2:
            delta = f"{((results['voc']/rated_voc)-1)*100:+.1f}%" if rated_voc else None
            st.metric("Voc", f"{results['voc']:.2f} V", delta=delta)
        with col3:
            delta = f"{((results['pmax']/rated_power)-1)*100:+.1f}%" if rated_power else None
            st.metric("Pmax", f"{results['pmax']:.1f} W", delta=delta)
        with col4:
            st.metric("Vmpp", f"{results['vmpp']:.2f} V")
        with col5:
            st.metric("Impp", f"{results['impp']:.3f} A")
        with col6:
            st.metric("Fill Factor", f"{results['ff']:.4f}")

        st.markdown("---")

        # Plots
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.subheader("I-V Characteristic")
            iv_df = pd.DataFrame({'Voltage (V)': voltage, 'Current (A)': current})
            st.line_chart(iv_df.set_index('Voltage (V)'))

        with col_plot2:
            st.subheader("P-V Characteristic")
            power = voltage * current
            pv_df = pd.DataFrame({'Voltage (V)': voltage, 'Power (W)': power})
            st.line_chart(pv_df.set_index('Voltage (V)'))

        st.markdown("---")

        # Resistance and quality
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.subheader("Resistance Parameters")
            st.metric("Series Resistance (Rs)", f"{results['rs']:.4f} Ω")
            st.metric("Shunt Resistance (Rsh)", f"{results['rsh']:.1f} Ω")

        with col_res2:
            st.subheader("Quality Indicators")

            if rated_power > 0:
                power_dev = ((results['pmax'] / rated_power) - 1) * 100
                if abs(power_dev) <= 3:
                    st.success(f"✅ Power within ±3% tolerance ({power_dev:+.2f}%)")
                elif abs(power_dev) <= 5:
                    st.warning(f"⚠️ Power deviation: {power_dev:+.2f}%")
                else:
                    st.error(f"❌ Power out of specification: {power_dev:+.2f}%")

            if results['ff'] >= 0.75:
                st.success(f"✅ Good fill factor: {results['ff']:.3f}")
            elif results['ff'] >= 0.70:
                st.warning(f"⚠️ Moderate fill factor: {results['ff']:.3f}")
            else:
                st.error(f"❌ Low fill factor: {results['ff']:.3f}")

        # Power binning
        if MODULES_LOADED:
            st.markdown("---")
            st.subheader("Power Bin Classification")
            bin_label, bin_lower, bin_upper = power_bin_classification(
                results['pmax'], rated_power, bin_size_w=5.0
            )
            col_bin1, col_bin2, col_bin3 = st.columns(3)
            with col_bin1:
                st.metric("Power Bin", bin_label)
            with col_bin2:
                st.metric("Bin Range Lower", f"{bin_lower:.0f} W")
            with col_bin3:
                upper_str = f"{bin_upper:.0f} W" if bin_upper != float('inf') else "∞"
                st.metric("Bin Range Upper", upper_str)
    else:
        st.info("📂 Please upload data or load demo data in the 'Data Upload' tab")

# Tab 3: STC Correction
with tab3:
    st.header("⚙️ STC Correction (IEC 60891)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Correction Procedure")

        procedure = st.selectbox(
            "Select Procedure",
            options=[1, 3, 4],
            format_func=lambda x: f"Procedure {x}"
        )

        if MODULES_LOADED:
            proc_info = CORRECTION_PROCEDURES[procedure]
            st.info(f"**{proc_info.name}**\n\n{proc_info.description}\n\n*Accuracy: {proc_info.accuracy}*")

        st.markdown("---")
        st.subheader("Temperature Coefficients")

        if MODULES_LOADED:
            tech_options = list(TYPICAL_TEMP_COEFFICIENTS.keys())
        else:
            tech_options = ["c-Si", "HJT", "TOPCon", "CdTe", "CIGS"]

        tech = st.selectbox("Module Technology", options=tech_options, index=0)

        # Default values
        alpha_default = 0.05
        beta_default = -0.30
        gamma_default = -0.40

        if MODULES_LOADED and tech in TYPICAL_TEMP_COEFFICIENTS:
            typical = TYPICAL_TEMP_COEFFICIENTS[tech]
            alpha_default = typical.alpha_isc_percent_per_c
            beta_default = typical.beta_voc_percent_per_c
            gamma_default = typical.gamma_pmax_percent_per_c

        alpha = st.number_input("Alpha - Isc (%/°C)", value=alpha_default, format="%.4f")
        beta = st.number_input("Beta - Voc (%/°C)", value=beta_default, format="%.4f")
        gamma = st.number_input("Gamma - Pmax (%/°C)", value=gamma_default, format="%.4f")

        if procedure in [1, 3]:
            rs = st.number_input("Rs (Ω)", value=0.30, format="%.4f")

        if procedure == 1:
            kappa = st.number_input("Kappa (Ω/°C)", value=0.001, format="%.5f")

    with col2:
        st.subheader("Correction Results")

        if st.session_state.iv_data is not None and st.session_state.analysis_results is not None and MODULES_LOADED:
            results = st.session_state.analysis_results
            voltage = st.session_state.iv_data['voltage']
            current = st.session_state.iv_data['current']

            if st.button("🔄 Apply STC Correction"):
                # Convert coefficients to absolute values
                alpha_abs = alpha / 100 * results['isc']
                beta_abs = beta / 100 * results['voc']

                if procedure == 1:
                    corrector = CorrectionProcedure1(alpha_abs, beta_abs, rs, kappa)
                elif procedure == 3:
                    corrector = CorrectionProcedure3(alpha_abs, beta_abs, rs)
                else:
                    corrector = CorrectionProcedure4(alpha_abs, beta_abs)

                v_stc, i_stc = corrector.correct_to_stc(
                    voltage, current, results['isc'],
                    measured_temp, measured_irrad
                )

                analyzer_stc = IVCurveAnalyzer(v_stc, i_stc)
                results_stc = analyzer_stc.extract_parameters()
                st.session_state.stc_results = results_stc

                st.markdown("### Comparison: Measured vs STC")

                comp_df = pd.DataFrame({
                    'Parameter': ['Isc (A)', 'Voc (V)', 'Pmax (W)', 'Vmpp (V)', 'Impp (A)', 'FF'],
                    'Measured': [
                        f"{results['isc']:.3f}", f"{results['voc']:.2f}",
                        f"{results['pmax']:.1f}", f"{results['vmpp']:.2f}",
                        f"{results['impp']:.3f}", f"{results['ff']:.4f}"
                    ],
                    'STC Corrected': [
                        f"{results_stc['isc']:.3f}", f"{results_stc['voc']:.2f}",
                        f"{results_stc['pmax']:.1f}", f"{results_stc['vmpp']:.2f}",
                        f"{results_stc['impp']:.3f}", f"{results_stc['ff']:.4f}"
                    ],
                    'Δ (%)': [
                        f"{((results_stc['isc']/results['isc'])-1)*100:+.2f}",
                        f"{((results_stc['voc']/results['voc'])-1)*100:+.2f}",
                        f"{((results_stc['pmax']/results['pmax'])-1)*100:+.2f}",
                        f"{((results_stc['vmpp']/results['vmpp'])-1)*100:+.2f}",
                        f"{((results_stc['impp']/results['impp'])-1)*100:+.2f}",
                        f"{((results_stc['ff']/results['ff'])-1)*100:+.2f}"
                    ]
                })

                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Plot comparison
                st.subheader("I-V Curve Comparison")
                comparison_df = pd.DataFrame({
                    'Voltage': np.concatenate([voltage, v_stc]),
                    'Current': np.concatenate([current, i_stc]),
                    'Type': ['Measured'] * len(voltage) + ['STC Corrected'] * len(v_stc)
                })
                st.line_chart(comparison_df.pivot(columns='Type', values='Current', index='Voltage'))
        else:
            st.info("📂 Please upload data first")

# Tab 4: Uncertainty Analysis
with tab4:
    st.header("📈 Uncertainty Analysis (GUM)")

    if st.session_state.analysis_results is not None and MODULES_LOADED:
        results = st.session_state.analysis_results

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pmax Uncertainty Budget")

            pmax_budget = create_pmax_uncertainty_budget(
                pmax_value=results['pmax'],
                repeatability_std=results['pmax'] * 0.003,
                n_measurements=3
            )

            pmax_result = pmax_budget.get_result()
            rel_unc = pmax_budget.get_relative_uncertainty()

            st.metric(
                "Pmax with Uncertainty",
                f"{results['pmax']:.1f} ± {pmax_result.expanded_uncertainty:.1f} W",
                delta=f"±{rel_unc[1]:.2f}% (k=2)"
            )

            # Budget table
            budget_table = pmax_budget.generate_budget_table()
            df = pd.DataFrame(budget_table)[['name', 'type', 'standard_uncertainty', 'contribution']]
            df.columns = ['Source', 'Type', 'Std Unc.', 'Contrib.']
            df['Std Unc.'] = df['Std Unc.'].apply(lambda x: f"{x:.4f}")
            df['Contrib.'] = df['Contrib.'].apply(lambda x: f"{x:.4f}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Uncertainty Summary")

            st.markdown(f"""
            | Quantity | Value |
            |----------|-------|
            | Combined Standard Uncertainty (u_c) | {pmax_result.standard_uncertainty:.3f} W |
            | Coverage Factor (k) | {pmax_result.coverage_factor:.2f} |
            | Expanded Uncertainty (U) | {pmax_result.expanded_uncertainty:.3f} W |
            | Relative Expanded Uncertainty | ±{rel_unc[1]:.2f}% |
            | Confidence Level | 95% |
            | Degrees of Freedom | {pmax_result.degrees_of_freedom} |
            """)

            st.markdown("---")
            st.subheader("Result Statement")
            st.success(f"**Pmax = {results['pmax']:.1f} ± {pmax_result.expanded_uncertainty:.1f} W** (k=2, 95% confidence)")
    else:
        st.info("📂 Please upload data or load demo data first")

# Tab 5: Reports
with tab5:
    st.header("📝 Test Reports")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Report Configuration")

        report_type = st.selectbox(
            "Report Type",
            ["IEC 60904-1 I-V Test Report", "Full Qualification Summary", "Quick Results"]
        )

        include_charts = st.checkbox("Include I-V/P-V Charts", value=True)
        include_uncertainty = st.checkbox("Include Uncertainty Budget", value=True)
        include_conformity = st.checkbox("Include Conformity Statement", value=True)

        st.markdown("---")
        st.subheader("Laboratory Information")
        lab_name = st.text_input("Laboratory Name", value="PV Testing Laboratory")
        report_number = st.text_input("Report Number", value=f"RPT-{datetime.now().strftime('%Y%m%d')}-001")
        operator = st.text_input("Operator", value="")

    with col2:
        st.subheader("Report Preview")

        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results

            report_text = f"""
# {report_type}
**Report Number:** {report_number}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Laboratory:** {lab_name}

## Test Conditions
- Temperature: {measured_temp}°C
- Irradiance: {measured_irrad} W/m²
- Equipment: {selected_equipment}

## Results at Measured Conditions

| Parameter | Value | Unit |
|-----------|-------|------|
| Isc | {results['isc']:.3f} | A |
| Voc | {results['voc']:.2f} | V |
| Pmax | {results['pmax']:.1f} | W |
| Vmpp | {results['vmpp']:.2f} | V |
| Impp | {results['impp']:.3f} | A |
| FF | {results['ff']:.4f} | - |
| Rs | {results['rs']:.4f} | Ω |
| Rsh | {results['rsh']:.1f} | Ω |
"""
            if include_conformity:
                power_dev = ((results['pmax'] / rated_power) - 1) * 100 if rated_power > 0 else 0
                conformity = "PASS" if abs(power_dev) <= 3 else "FAIL"
                report_text += f"""
## Conformity Statement
The module **{conformity}ES** the power rating specification of {rated_power:.0f}W ±3%.
Measured deviation: {power_dev:+.2f}%
"""

            st.markdown(report_text)

            st.markdown("---")
            st.subheader("Export Options")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                st.download_button(
                    "📄 Download TXT",
                    data=report_text,
                    file_name=f"{report_number}.txt",
                    mime="text/plain"
                )

            with col_exp2:
                csv_data = pd.DataFrame([{
                    'Report': report_number,
                    'Date': datetime.now().isoformat(),
                    'Isc_A': results['isc'],
                    'Voc_V': results['voc'],
                    'Pmax_W': results['pmax'],
                    'Vmpp_V': results['vmpp'],
                    'Impp_A': results['impp'],
                    'FF': results['ff'],
                    'Rs_Ohm': results['rs'],
                    'Rsh_Ohm': results['rsh']
                }]).to_csv(index=False)

                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=f"{report_number}.csv",
                    mime="text/csv"
                )

            with col_exp3:
                st.button("📑 Generate PDF", disabled=True, help="PDF export coming soon")
        else:
            st.info("📂 Please upload data first to generate reports")

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
    **🔬 Analysis Modules**
    I-V Curve • STC Correction • Uncertainty • Temp Coeff
    """)

with col3:
    st.markdown("""
    **© 2025 Solar PV Test Platform**
    ISO/IEC 17025:2017
    """)
