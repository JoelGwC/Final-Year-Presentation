import streamlit as st
import numpy as np
import subprocess
import os

# Configure the page
st.set_page_config(page_title="RAFFC OpAmp EDA", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# Main Title
st.title("RAFFC Frequency Compensation EDA Tool")

# --- 1. SCHEMATIC DISPLAY ---
st.header("1. Circuit Schematic")
image_path = "RAFFC_schematic.png"
if os.path.exists(image_path):
    st.image(image_path, caption="RAFFC OpAmp Schematic", use_container_width=True)
else:
    st.info(f"Schematic image '{image_path}' not found. Please save your image in this folder to display it.")

# --- 2. INTERACTIVE WIDGET ---
st.header("2. Interactive Compensation Parameters")
st.markdown("Explore the mathematical relationships for RAFFC active compensation instantly.")

# Create columns for the layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Design Variables")
    gm1 = st.slider("gm1 (uA/V) - First Stage", min_value=10.0, max_value=500.0, value=140.0, step=10.0)
    gmb = st.slider("gmb (uA/V) - RAFFC Active Feedback", min_value=10.0, max_value=500.0, value=280.0, step=10.0)
    gm3 = st.slider("gm3 (uA/V) - Third Stage", min_value=10.0, max_value=1000.0, value=390.0, step=10.0)
    load_cl = st.slider("Load CL (pF)", min_value=50.0, max_value=1000.0, value=500.0, step=10.0)

# Convert to SI units for calculations
gm1_si = gm1 * 1e-6
gmb_si = gmb * 1e-6
gm3_si = gm3 * 1e-6
cl_si = load_cl * 1e-12
cc1_si = 11e-12 # Fixed 11pF as per circuit.py

# Mathematical evaluations
is_stable = gmb_si > gm1_si
gbw_hz = gm1_si / (2 * np.pi * cc1_si)
gbw_mhz = gbw_hz / 1e6

ratio = gmb_si / max(gm1_si, 1e-12)
ideal_pm_rad = np.arctan((ratio**3) / (ratio**2 + 2))
ideal_pm_deg = np.degrees(ideal_pm_rad)

cc2_req = (2 * gm3_si * (cc1_si**2)) / (gmb_si * cl_si)
cc2_req_pf = cc2_req * 1e12

with col_right:
    st.subheader("Live Performance Metrics")
    
    # Custom HTML for the Stability Status box
    status_color = "#22c55e" if is_stable else "#ef4444"
    status_text = "STABLE (gmb > gm1)" if is_stable else "UNSTABLE (gmb <= gm1)"
    
    st.markdown(f"""
    <div style="border: 2px solid {status_color}; border-radius: 8px; padding: 20px; margin-bottom: 20px; background-color: rgba(0,0,0,0.2);">
        <h3 style="margin-top:0px; color: {status_color};">Stability Status</h3>
        <p style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{status_text}</p>
        <p style="margin: 0px; color: #a3a3a3;">Target CC2: {cc2_req_pf:.2f} pF</p>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("GBW", f"{gbw_mhz:.2f} MHz")
    with m2:
        st.metric("Phase Margin", f"{ideal_pm_deg:.1f}°")
    with m3:
        st.metric("Req. CC2", f"{cc2_req_pf:.2f} pF")


# --- 3. OPTIMIZER ---
st.markdown("---")
st.header("3. NSGA-II Dimension Optimizer")
st.markdown("Run the full ANN-based NSGA2 optimizer to synthesize physical transistor dimensions.")

if st.button("🚀 Run Optimizer", type="primary"):
    st.info("Running optimizer... This may take a minute depending on your hardware.")
    
    output_placeholder = st.empty()
    full_output = ""
    
    try:
        process = subprocess.Popen(
            ["python", "optimizer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream the output line by line into the UI
        for line in process.stdout:
            full_output += line
            output_placeholder.code(full_output, language="text")
            
        process.wait()
        
        if process.returncode == 0:
            st.success("Optimization completed successfully!")
            st.balloons()
        else:
            st.error("Optimization failed. Check the logs above for errors.")
            
    except Exception as e:
        st.error(f"Failed to execute optimizer: {e}")
