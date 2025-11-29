#!/usr/bin/env python3
"""Demo script to test the Solar PV Data Analysis Platform.

Demonstrates:
1. Loading I-V data
2. Parameter extraction
3. STC correction using different procedures
"""

import numpy as np
from src.ingestion.auto_detector import AutoDetector
from src.analysis.iv_curve import IVCurveAnalyzer
from src.analysis.corrections import CorrectionProcedure1, CorrectionProcedure4


def generate_sample_iv_data():
    """Generate sample I-V data for testing."""
    # Typical c-Si module at 800 W/m² and 30°C
    voltage = np.linspace(0, 55, 100)
    
    # Single-diode equation approximation
    v_oc = 52.0
    i_sc = 12.5
    ff = 0.78
    
    # Simplified I-V characteristic
    current = i_sc * (1 - (voltage / v_oc)**3) * np.exp(-voltage / (v_oc * 0.6))
    current = np.maximum(current, 0)  # No negative current
    
    return voltage, current


def main():
    print("=" * 70)
    print("Solar PV Data Analysis Platform - Demo")
    print("IEC 60904, 60891, 61215 Compliant")
    print("=" * 70)
    print()
    
    # Generate sample data
    print("📊 Generating sample I-V data...")
    voltage, current = generate_sample_iv_data()
    print(f"   Generated {len(voltage)} data points")
    print()
    
    # Measured conditions
    measured_temp = 30.0  # °C
    measured_irrad = 800.0  # W/m²
    
    print(f"🌡️ Measured Conditions:")
    print(f"   Temperature: {measured_temp}°C")
    print(f"   Irradiance: {measured_irrad} W/m²")
    print()
    
    # Analyze I-V curve
    print("🔬 Analyzing I-V curve...")
    analyzer = IVCurveAnalyzer(voltage, current)
    results = analyzer.extract_parameters()
    
    print("\n" + "="*70)
    print("RAW MEASUREMENT RESULTS (at measured conditions)")
    print("="*70)
    print(f"Short-Circuit Current (Isc):  {results['isc']:.3f} A")
    print(f"Open-Circuit Voltage (Voc):   {results['voc']:.3f} V")
    print(f"Maximum Power (Pmax):         {results['pmax']:.3f} W")
    print(f"Voltage at MPP (Vmpp):        {results['vmpp']:.3f} V")
    print(f"Current at MPP (Impp):        {results['impp']:.3f} A")
    print(f"Fill Factor (FF):             {results['ff']:.4f}")
    print(f"Series Resistance (Rs):       {results['rs']:.4f} Ω")
    print(f"Shunt Resistance (Rsh):       {results['rsh']:.1f} Ω")
    print()
    
    # Apply STC correction using Procedure 1
    print("\n" + "="*70)
    print("APPLYING IEC 60891 PROCEDURE 1 (Full Correction)")
    print("="*70)
    
    # Typical c-Si temperature coefficients
    alpha = 0.0005  # A/°C
    beta = -0.003   # V/°C
    rs = results['rs']
    kappa = 0.001   # Ω/°C
    
    print(f"Temperature Coefficients:")
    print(f"   Alpha (Isc): {alpha:.5f} A/°C")
    print(f"   Beta (Voc):  {beta:.5f} V/°C")
    print(f"   Rs:          {rs:.4f} Ω")
    print(f"   Kappa:       {kappa:.5f} Ω/°C")
    print()
    
    corrector = CorrectionProcedure1(alpha, beta, rs, kappa)
    v_stc, i_stc = corrector.correct_to_stc(
        voltage, current, results['isc'],
        measured_temp, measured_irrad
    )
    
    # Analyze corrected curve
    analyzer_stc = IVCurveAnalyzer(v_stc, i_stc)
    results_stc = analyzer_stc.extract_parameters()
    
    print("\n" + "="*70)
    print("CORRECTED RESULTS (at STC: 25°C, 1000 W/m²)")
    print("="*70)
    print(f"Short-Circuit Current (Isc):  {results_stc['isc']:.3f} A  [Change: {((results_stc['isc']/results['isc'])-1)*100:+.2f}%]")
    print(f"Open-Circuit Voltage (Voc):   {results_stc['voc']:.3f} V  [Change: {((results_stc['voc']/results['voc'])-1)*100:+.2f}%]")
    print(f"Maximum Power (Pmax):         {results_stc['pmax']:.3f} W  [Change: {((results_stc['pmax']/results['pmax'])-1)*100:+.2f}%]")
    print(f"Voltage at MPP (Vmpp):        {results_stc['vmpp']:.3f} V")
    print(f"Current at MPP (Impp):        {results_stc['impp']:.3f} A")
    print(f"Fill Factor (FF):             {results_stc['ff']:.4f}")
    print()
    
    # Compare with Procedure 4 (quick correction)
    print("\n" + "="*70)
    print("COMPARISON: Procedure 4 (Quick Correction)")
    print("="*70)
    
    corrector4 = CorrectionProcedure4(alpha, beta)
    v_stc4, i_stc4 = corrector4.correct_to_stc(
        voltage, current, results['isc'],
        measured_temp, measured_irrad
    )
    
    analyzer_stc4 = IVCurveAnalyzer(v_stc4, i_stc4)
    results_stc4 = analyzer_stc4.extract_parameters()
    
    print(f"Pmax (Procedure 1): {results_stc['pmax']:.3f} W")
    print(f"Pmax (Procedure 4): {results_stc4['pmax']:.3f} W")
    print(f"Difference:         {abs(results_stc['pmax'] - results_stc4['pmax']):.3f} W ({abs((results_stc4['pmax']/results_stc['pmax'])-1)*100:.2f}%)")
    print()
    
    print("=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)
    print()
    print("🚀 Next Steps:")
    print("   1. Run Streamlit dashboard: streamlit run app.py")
    print("   2. Upload your real test data (.txt, .csv, .xlsx)")
    print("   3. Analyze and generate reports")
    print()


if __name__ == "__main__":
    main()
