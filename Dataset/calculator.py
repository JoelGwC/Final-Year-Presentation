import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np

def calculate_raffc(gm1_uA, gm3_uA, gmb_uA, CL_pF, CC1_pF):
    # Convert to standard units
    gm1, gm3, gmb = gm1_uA * 1e-6, gm3_uA * 1e-6, gmb_uA * 1e-6
    CL, CC1 = CL_pF * 1e-12, CC1_pF * 1e-12
    
    # Mathematical Equations from Grasso et al.
    gbw_mhz = (gm1 / (2 * np.pi * CC1)) / 1e6
    
    ratio = gmb / gm1
    pm_deg = np.degrees(np.arctan((ratio**3) / (ratio**2 + 2)))
    
    cc2_req_pf = ((2 * gm3 * (CC1**2)) / (gmb * CL)) * 1e12
    
    # Display Formatting
    html_out = f"""
    <div style='background-color: #f0f4f8; padding: 20px; border-radius: 8px; font-family: Arial;'>
        <h3 style='margin-top:0;'>RAFFC Output Metrics</h3>
        <p><b>Gain-Bandwidth Product (GBW):</b> {gbw_mhz:.2f} MHz</p>
        <p><b>Phase Margin (PM):</b> {pm_deg:.2f}°</p>
        <p><b>Required CC2:</b> {cc2_req_pf:.3f} pF</p>
    """
    
    if gm1 >= gmb:
        html_out += f"<p style='color: red; font-weight: bold;'>⚠️ WARNING: Asymptotic Stability Violated (gmb must be > gm1)</p>"
        
    html_out += "</div>"
    display(HTML(html_out))

# Define Interactive Sliders
style = {'description_width': 'initial'}
layout = widgets.Layout(width='500px')

gm1_slider = widgets.FloatSlider(value=140, min=50, max=500, step=1, description='First Stage gm1 (uA/V):', style=style, layout=layout)
gm3_slider = widgets.FloatSlider(value=390, min=100, max=1000, step=1, description='Third Stage gm3 (uA/V):', style=style, layout=layout)
gmb_slider = widgets.FloatSlider(value=280, min=50, max=800, step=1, description='Feedback gmb (uA/V):', style=style, layout=layout)
cl_slider = widgets.FloatSlider(value=500, min=10, max=1000, step=10, description='Load Capacitance CL (pF):', style=style, layout=layout)
cc1_slider = widgets.FloatSlider(value=11, min=1, max=50, step=0.5, description='Primary Comp CC1 (pF):', style=style, layout=layout)

# Link sliders to function
ui = widgets.VBox([gm1_slider, gm3_slider, gmb_slider, cl_slider, cc1_slider])
out = widgets.interactive_output(    calculate_raffc, {
        'gm1_uA': gm1_slider,
        'gm3_uA': gm3_slider,
        'gmb_uA': gmb_slider,
        'CL_pF': cl_slider,
        'CC1_pF': cc1_slider
    })

display(ui, out)