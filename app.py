"""Solar PV Data Analysis Platform - Streamlit Dashboard.

Comprehensive I-V curve analysis, STC corrections, and reporting
per IEC 60904, 60891, 61215, and 61853 standards.
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
    from src.analysis.hotspot_detection import (
        HotspotDetector,
        CellConfiguration,
        detect_hotspots,
    )
    from src.analysis.energy_rating import (
        EnergyRatingCalculator,
        ClimateProfile,
        CLIMATE_PROFILES,
        create_sample_power_matrix,
        calculate_energy_rating,
    )
    from src.analysis.iam_analysis import (
        IAMAnalyzer,
        IAMModel,
        IAMParameters,
    )
    from src.analysis.bifaciality import (
        BifacialAnalyzer,
        MountingConfiguration,
        estimate_bifacial_gain,
        get_ground_albedo,
    )
    from config.equipment_registry import list_all_equipment, get_equipment
    from config.iec_standards import TYPICAL_TEMP_COEFFICIENTS, CORRECTION_PROCEDURES
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)

# Initialize database and seed protocols
DB_INITIALIZED = False
DB_INIT_ERROR = None
try:
    from src.database import init_database, get_db, Protocol
    # Initialize database with protocol seeding
    init_database()
    DB_INITIALIZED = True
    print("Database initialized and protocols seeded successfully")
except Exception as e:
    DB_INIT_ERROR = str(e)
    print(f"Database initialization skipped or failed: {e}")

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
if 'cell_temps' not in st.session_state:
    st.session_state.cell_temps = None


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


def generate_sample_cell_temperatures(config: CellConfiguration = CellConfiguration.CELLS_72):
    """Generate sample cell temperature data with hotspots."""
    if config == CellConfiguration.CELLS_72:
        rows, cols = 12, 6
    elif config == CellConfiguration.CELLS_60:
        rows, cols = 10, 6
    elif config == CellConfiguration.CELLS_144:
        rows, cols = 24, 6
    else:
        rows, cols = 12, 6

    # Base temperature with some variation
    base_temp = 45.0
    temps = base_temp + np.random.normal(0, 2, (rows, cols))

    # Add a few hotspots
    temps[3, 2] += 15  # Hotspot
    temps[7, 4] += 12  # Warm spot
    temps[2, 1] += 8   # Warm spot

    # Add a weak cell (cooler)
    temps[9, 3] -= 8

    return temps


def create_iv_plot(voltage, current, results=None):
    """Create I-V and P-V curves using Streamlit native charts."""
    power = voltage * current

    # Create dataframes for plotting
    iv_df = pd.DataFrame({'Voltage (V)': voltage, 'Current (A)': current})
    pv_df = pd.DataFrame({'Voltage (V)': voltage, 'Power (W)': power})

    return iv_df, pv_df


# Main header
st.markdown('<div class="main-header">☀️ Solar PV Data Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IEC 60904, 60891, 61215, 61853 Compliant Testing & Analysis</div>', unsafe_allow_html=True)

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
    module_area = st.number_input("Module Area (m²)", value=2.56, step=0.01)

    st.markdown("---")
    st.markdown("### 📚 Standards")
    st.markdown("""
    - IEC 60904-1: I-V Curves
    - IEC 60891: STC Correction
    - IEC 60904-9: Simulator Class
    - IEC 60904-7: Spectral Match
    - IEC 61853: Energy Rating
    - IEC TS 60904-1-2: Bifacial
    """)

    st.markdown("---")
    st.info("📦 Version 3.0.0")

# Check if modules loaded
if not MODULES_LOADED:
    st.warning(f"Some modules could not be loaded: {IMPORT_ERROR}. Running in demo mode.")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📂 Data Upload", "📊 I-V Analysis", "⚙️ STC Correction",
    "📈 Uncertainty", "🔥 Hotspot Analysis", "⚡ Energy Rating",
    "📐 IAM Analysis", "🔄 Bifaciality", "📝 Reports"
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

# Tab 5: Hotspot Analysis
with tab5:
    st.header("🔥 Hotspot Detection & Cell Analysis")

    if MODULES_LOADED:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            cell_config = st.selectbox(
                "Cell Configuration",
                options=[
                    CellConfiguration.CELLS_60,
                    CellConfiguration.CELLS_72,
                    CellConfiguration.CELLS_120,
                    CellConfiguration.CELLS_144
                ],
                format_func=lambda x: x.value,
                index=1
            )

            hot_threshold = st.slider("Hot Threshold (°C above mean)", 5.0, 20.0, 10.0)
            hotspot_threshold = st.slider("Hotspot Threshold (°C above mean)", 10.0, 30.0, 15.0)

            st.markdown("---")
            if st.button("🔬 Generate Sample Temperature Data"):
                st.session_state.cell_temps = generate_sample_cell_temperatures(cell_config)
                st.success("✅ Sample temperature data generated!")
                st.rerun()

        with col2:
            st.subheader("Thermal Analysis Results")

            if st.session_state.cell_temps is not None:
                temps = st.session_state.cell_temps

                # Run hotspot detection
                detector = HotspotDetector(
                    configuration=cell_config,
                    hot_threshold=hot_threshold,
                    hotspot_threshold=hotspot_threshold
                )
                result = detector.analyze(temps)

                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Mean Temp", f"{result.mean_temperature:.1f}°C")
                with col_m2:
                    st.metric("Max ΔT", f"{result.max_delta_t:.1f}°C")
                with col_m3:
                    st.metric("Hotspots", result.hotspot_count)
                with col_m4:
                    st.metric("Uniformity", f"{result.uniformity_index:.1f}%")

                # Status indicator
                status_colors = {
                    'healthy': 'success',
                    'minor_issues': 'warning',
                    'attention_needed': 'warning',
                    'critical': 'error'
                }
                status_func = getattr(st, status_colors.get(result.overall_status, 'info'))
                status_func(f"Overall Status: **{result.overall_status.upper()}**")

                # Heatmap visualization
                st.subheader("Cell Temperature Heatmap")

                # Create DataFrame for heatmap
                temp_df = pd.DataFrame(temps)
                temp_df.index = [f"Row {i+1}" for i in range(temps.shape[0])]
                temp_df.columns = [f"Col {j+1}" for j in range(temps.shape[1])]

                # Display as styled table (color-coded)
                def color_temp(val):
                    mean_t = result.mean_temperature
                    if val > mean_t + hotspot_threshold:
                        return 'background-color: #ff4444; color: white'
                    elif val > mean_t + hot_threshold:
                        return 'background-color: #ff8844'
                    elif val < mean_t - 10:
                        return 'background-color: #4488ff'
                    else:
                        return 'background-color: #88cc88'

                styled_df = temp_df.style.applymap(color_temp).format("{:.1f}°C")
                st.dataframe(styled_df, use_container_width=True)

                # String analysis
                st.markdown("---")
                st.subheader("String Analysis")

                string_data = []
                for s in result.string_analysis:
                    string_data.append({
                        'String': f"String {s.string_id + 1}",
                        'Mean Temp (°C)': f"{s.mean_temp:.1f}",
                        'Std Dev (°C)': f"{s.std_temp:.2f}",
                        'Hotspots': s.n_hotspots,
                        'Weak Cells': s.n_weak_cells,
                        'Mismatch': s.mismatch_severity.upper()
                    })

                string_df = pd.DataFrame(string_data)
                st.dataframe(string_df, use_container_width=True, hide_index=True)

                # Hotspot details
                if result.hotspot_cells:
                    st.markdown("---")
                    st.subheader("Hotspot Details")
                    for h in result.hotspot_cells[:5]:
                        st.warning(f"Cell ({h.row+1}, {h.col+1}): {h.temperature:.1f}°C (ΔT: {h.delta_t:+.1f}°C, Z-score: {h.z_score:.2f})")
            else:
                st.info("👆 Click 'Generate Sample Temperature Data' to run analysis")
    else:
        st.warning("Hotspot detection module not available")

# Tab 6: Energy Rating
with tab6:
    st.header("⚡ Energy Rating (IEC 61853)")

    if MODULES_LOADED:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Module Parameters")

            pmax_stc = st.number_input("Pmax STC (W)", value=rated_power, key="er_pmax")
            er_module_area = st.number_input("Module Area (m²)", value=module_area, key="er_area")
            temp_coeff_pmax = st.number_input("Temp Coeff Pmax (%/°C)", value=-0.35, format="%.3f")

            st.markdown("---")
            st.subheader("Climate Profile")

            climate = st.selectbox(
                "Reference Climate",
                options=list(ClimateProfile),
                format_func=lambda x: CLIMATE_PROFILES[x].name
            )

            climate_data = CLIMATE_PROFILES[climate]
            st.info(f"**{climate_data.description}**\n\nAnnual GHI: {climate_data.annual_irradiation} kWh/m²")

            st.markdown("---")
            st.subheader("Correction Factors")
            spectral_factor = st.slider("Spectral Factor", 0.95, 1.05, 1.0)
            iam_factor = st.slider("IAM Factor", 0.90, 1.0, 0.97)
            soiling_factor = st.slider("Soiling Factor", 0.90, 1.0, 0.98)

        with col2:
            st.subheader("Energy Rating Results")

            if st.button("📊 Calculate Energy Rating"):
                # Create power matrix and calculator
                matrix = create_sample_power_matrix(pmax_stc, temp_coeff_pmax)
                calculator = EnergyRatingCalculator(
                    matrix, er_module_area, pmax_stc, temp_coeff_pmax
                )

                # Calculate CSER
                result = calculator.calculate_cser(
                    climate,
                    spectral_factor=spectral_factor,
                    iam_factor=iam_factor,
                    soiling_factor=soiling_factor
                )

                # Display results
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("CSER", f"{result.cser:.1f} kWh")
                with col_r2:
                    st.metric("CSPR", f"{result.cspr:.1f} W")
                with col_r3:
                    st.metric("PR", f"{result.performance_ratio:.1%}")

                col_r4, col_r5, col_r6 = st.columns(3)
                with col_r4:
                    st.metric("Specific Yield", f"{result.annual_energy:.0f} kWh/kWp")
                with col_r5:
                    st.metric("Temp Loss", f"{result.temperature_loss:.1f}%")
                with col_r6:
                    st.metric("Low-Irr Loss", f"{result.low_irradiance_loss:.1f}%")

                st.markdown("---")
                st.subheader("Efficiency Curves")

                # Generate efficiency curve
                irr, eff = calculator.generate_efficiency_curve(25.0)
                eff_df = pd.DataFrame({
                    'Irradiance (W/m²)': irr,
                    'Efficiency (%)': eff
                })
                st.line_chart(eff_df.set_index('Irradiance (W/m²)'))

                # Temperature derating curve
                st.subheader("Temperature Derating")
                temps, rel_power = calculator.generate_temperature_derating_curve(1000.0)
                temp_df = pd.DataFrame({
                    'Temperature (°C)': temps,
                    'Relative Power': rel_power
                })
                st.line_chart(temp_df.set_index('Temperature (°C)'))

                # Summary table
                st.markdown("---")
                st.subheader("Rating Summary")
                summary_df = pd.DataFrame({
                    'Parameter': [
                        'Climate Profile',
                        'Annual Irradiation',
                        'Climate-Specific Energy Rating (CSER)',
                        'Climate-Specific Power Rating (CSPR)',
                        'Annual Specific Yield',
                        'Performance Ratio',
                        'Temperature Losses',
                        'Low Irradiance Losses'
                    ],
                    'Value': [
                        result.climate_profile,
                        f"{result.reference_yield:.0f} kWh/m²/year",
                        f"{result.cser:.1f} kWh",
                        f"{result.cspr:.1f} W",
                        f"{result.annual_energy:.0f} kWh/kWp",
                        f"{result.performance_ratio:.1%}",
                        f"{result.temperature_loss:.1f}%",
                        f"{result.low_irradiance_loss:.1f}%"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Energy rating module not available")

# Tab 7: IAM Analysis
with tab7:
    st.header("📐 Incidence Angle Modifier (IAM) Analysis")

    if MODULES_LOADED:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Model Selection")

            iam_model = st.selectbox(
                "IAM Model",
                options=[IAMModel.ASHRAE, IAMModel.MARTIN_RUIZ, IAMModel.PHYSICAL],
                format_func=lambda x: x.value.upper()
            )

            st.markdown("---")
            st.subheader("Model Parameters")

            if iam_model == IAMModel.ASHRAE:
                b0 = st.slider("b₀ Parameter", 0.02, 0.10, 0.05)
                params = IAMParameters(model=iam_model, b0=b0)
            elif iam_model == IAMModel.MARTIN_RUIZ:
                ar = st.slider("aᵣ Parameter", 0.10, 0.30, 0.16)
                params = IAMParameters(model=iam_model, ar=ar)
            else:
                n_glass = st.slider("Glass Refractive Index", 1.40, 1.60, 1.526)
                params = IAMParameters(model=iam_model, n_glass=n_glass)

            st.markdown("---")
            st.subheader("Site Parameters")
            latitude = st.slider("Latitude (°)", -60, 60, 40)
            tilt = st.slider("Array Tilt (°)", 0, 90, 25)
            azimuth = st.slider("Array Azimuth (°)", 90, 270, 180)

        with col2:
            st.subheader("IAM Curve")

            analyzer = IAMAnalyzer(iam_model)
            angles, iam_values = analyzer.get_iam_curve(params)

            # Plot IAM curve
            iam_df = pd.DataFrame({
                'Angle (°)': angles,
                'IAM': iam_values
            })
            st.line_chart(iam_df.set_index('Angle (°)'))

            # Key values
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                iam_0 = analyzer.calculate_iam(0, params)
                st.metric("IAM at 0°", f"{iam_0.iam:.4f}")
            with col_k2:
                iam_30 = analyzer.calculate_iam(30, params)
                st.metric("IAM at 30°", f"{iam_30.iam:.4f}")
            with col_k3:
                iam_60 = analyzer.calculate_iam(60, params)
                st.metric("IAM at 60°", f"{iam_60.iam:.4f}")
            with col_k4:
                iam_80 = analyzer.calculate_iam(80, params)
                st.metric("IAM at 80°", f"{iam_80.iam:.4f}")

            st.markdown("---")
            st.subheader("Annual Angular Losses")

            losses = analyzer.calculate_annual_angular_loss(latitude, tilt, azimuth, params)

            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.metric("Average IAM", f"{losses['average_iam']:.4f}")
            with col_l2:
                st.metric("Annual Loss", f"{losses['annual_angular_loss_percent']:.2f}%")
            with col_l3:
                st.metric("Adjusted Loss", f"{losses['adjusted_loss_percent']:.2f}%")

            # Model comparison
            st.markdown("---")
            st.subheader("Model Comparison")

            comparison = analyzer.compare_models()
            comp_df = pd.DataFrame({
                'Angle': comparison['angles'],
                'ASHRAE (b₀=0.05)': comparison['ashrae_b0_05'],
                'Martin-Ruiz (aᵣ=0.16)': comparison['martin_ruiz_ar_16'],
                'Physical (n=1.526)': comparison['physical_n_1526']
            })
            st.line_chart(comp_df.set_index('Angle'))
    else:
        st.warning("IAM analysis module not available")

# Tab 8: Bifaciality
with tab8:
    st.header("🔄 Bifaciality Analysis (IEC TS 60904-1-2)")

    if MODULES_LOADED:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Module Parameters")

            technology = st.selectbox(
                "Cell Technology",
                options=['mono_perc', 'mono_topcon', 'mono_hjt', 'multi_perc', 'n_type_pert'],
                format_func=lambda x: x.replace('_', ' ').upper()
            )

            typical = BifacialAnalyzer.TYPICAL_BIFACIALITY.get(technology, {})
            st.info(f"Typical φ_Pmax: {typical.get('phi_pmax', 0.70):.2f}\nRange: {typical.get('range', (0.65, 0.85))}")

            phi_pmax = st.slider("Bifaciality Factor (φ)", 0.50, 1.00, typical.get('phi_pmax', 0.70))

            st.markdown("---")
            st.subheader("Installation Configuration")

            mounting = st.selectbox(
                "Mounting Type",
                options=list(MountingConfiguration),
                format_func=lambda x: x.value.replace('_', ' ').title()
            )

            ground_type = st.selectbox(
                "Ground Surface",
                options=['grass', 'concrete', 'sand', 'white_gravel', 'dark_soil', 'snow']
            )
            albedo = get_ground_albedo(ground_type)
            st.info(f"Ground Albedo: {albedo:.2f}")

            gcr = st.slider("Ground Coverage Ratio", 0.20, 0.60, 0.40)
            height = st.slider("Module Height (m)", 0.5, 5.0, 1.5)

        with col2:
            st.subheader("Bifacial Gain Analysis")

            analyzer = BifacialAnalyzer(
                module_area=module_area,
                pmax_stc_front=rated_power,
                technology=technology
            )

            # Calculate annual bifacial gain
            annual_result = analyzer.annual_bifacial_gain(
                phi_pmax=phi_pmax,
                albedo=albedo,
                gcr=gcr,
                height=height,
                mounting=mounting
            )

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Annual BG", f"{annual_result['annual_bifacial_gain_percent']:.1f}%")
            with col_r2:
                st.metric("Energy Gain", f"{annual_result['energy_weighted_gain_percent']:.1f}%")
            with col_r3:
                st.metric("Rear/Front Ratio", f"{annual_result['average_rear_irradiance_ratio']:.1%}")

            st.markdown("---")
            st.subheader("Irradiance Scenario Analysis")

            # Scenario analysis
            front_irr = st.slider("Front Irradiance (W/m²)", 200, 1200, 1000)

            scenarios = []
            for rear_ratio in [0.05, 0.10, 0.15, 0.20, 0.25]:
                rear_irr = front_irr * rear_ratio
                gain_result = analyzer.calculate_bifacial_gain(
                    front_irr, rear_irr, phi_pmax
                )
                scenarios.append({
                    'Rear/Front Ratio': f"{rear_ratio:.0%}",
                    'Rear Irradiance (W/m²)': f"{rear_irr:.0f}",
                    'Power Mono (W)': f"{gain_result.power_monofacial:.1f}",
                    'Power Bifi (W)': f"{gain_result.power_bifacial:.1f}",
                    'Bifacial Gain': f"{gain_result.bifacial_gain:.1f}%"
                })

            scenario_df = pd.DataFrame(scenarios)
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

            # Rear irradiance estimation
            st.markdown("---")
            st.subheader("Rear Irradiance Estimation")

            ghi = st.number_input("GHI (W/m²)", value=800.0)
            dhi = st.number_input("DHI (W/m²)", value=150.0)
            dni = ghi - dhi  # Simplified
            solar_elev = st.slider("Solar Elevation (°)", 10, 80, 45)

            rear_est = analyzer.estimate_rear_irradiance(
                ghi=ghi,
                dni=dni,
                dhi=dhi,
                albedo=albedo,
                height=height,
                gcr=gcr,
                solar_elevation=solar_elev
            )

            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric("Total Rear", f"{rear_est['rear_total']:.1f} W/m²")
            with col_e2:
                st.metric("From Ground", f"{rear_est['rear_from_ground']:.1f} W/m²")
            with col_e3:
                st.metric("Rear/Front", f"{rear_est['rear_front_ratio']:.1%}")

            st.info(f"**Optimal Height:** {annual_result['optimal_height_m']:.1f}m for this mounting configuration")
    else:
        st.warning("Bifaciality module not available")

# Tab 9: Reports
with tab9:
    st.header("📝 Test Reports")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Report Configuration")

        report_type = st.selectbox(
            "Report Type",
            ["IEC 60904-1 I-V Test Report", "Full Qualification Summary", "Energy Rating Report", "Quick Results"]
        )

        include_charts = st.checkbox("Include I-V/P-V Charts", value=True)
        include_uncertainty = st.checkbox("Include Uncertainty Budget", value=True)
        include_conformity = st.checkbox("Include Conformity Statement", value=True)
        include_energy = st.checkbox("Include Energy Rating", value=False)

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
    IEC TS 60904-1-2
    """)

with col2:
    st.markdown("""
    **🔬 Analysis Modules**
    I-V Curve • STC Correction • Uncertainty
    Hotspot • Energy Rating • IAM • Bifacial
    """)

with col3:
    st.markdown("""
    **© 2025 Solar PV Test Platform**
    ISO/IEC 17025:2017
    """)
