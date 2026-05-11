import torch
import numpy as np
import joblib
from model import SurrogateModel 

class RAFFC_OpAmp:
    def __init__(self, nmos_pth, pmos_pth, scaler_X_nmos_path, scaler_y_nmos_path, scaler_X_pmos_path, scaler_y_pmos_path):
        # Load scalers to normalize inputs/outputs for the ANN
        self.scaler_X_nmos = joblib.load(scaler_X_nmos_path)
        self.scaler_y_nmos = joblib.load(scaler_y_nmos_path)
        self.scaler_X_pmos = joblib.load(scaler_X_pmos_path)
        self.scaler_y_pmos = joblib.load(scaler_y_pmos_path)
        
        # Load the trained Surrogate Models (ANNs)
        self.nmos_model = SurrogateModel()
        self.nmos_model.load_state_dict(torch.load(nmos_pth))
        self.nmos_model.eval()
        
        self.pmos_model = SurrogateModel()
        self.pmos_model.load_state_dict(torch.load(pmos_pth))
        self.pmos_model.eval()
        
        # Global Circuit Specifications (from the paper's test bench)
        self.VDD = 1.0       # Volts
        self.CL = 500e-12    # 500 pF Load Capacitance
        self.CC1 = 11e-12    # 11 pF Primary Compensation Capacitor

    def get_transistor_params(self, gm_id, L, vds, is_nmos=True):
        """Queries the ANN to unpack all 6 Day 1 physical parameters."""
        raw_inputs = np.array([[gm_id, L, vds]])
        
        with torch.no_grad():
            if is_nmos:
                scaled_inputs = self.scaler_X_nmos.transform(raw_inputs)
                tensor_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
                scaled_preds = self.nmos_model(tensor_inputs).numpy()
                real_preds = self.scaler_y_nmos.inverse_transform(scaled_preds)
            else:
                scaled_inputs = self.scaler_X_pmos.transform(raw_inputs)
                tensor_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
                scaled_preds = self.pmos_model(tensor_inputs).numpy()
                real_preds = self.scaler_y_pmos.inverse_transform(scaled_preds)
                
        # CRITICAL UPDATE: Unpack all 6 physical targets cleanly
        id_w, gds_w, vgs, vdsat, cgg_w, cdd_w = real_preds[0]
        
        return id_w, gds_w, vgs, vdsat, cgg_w, cdd_w

    def evaluate(self, sizing_guesses):
        """
        sizing_guesses structure: 
        [gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b, Id_1, Id_2, Id_3, Id_b]
        """
        # --- 1. EXTRACT OPTIMIZER GUESSES ---
        gm_id_1, L_1 = sizing_guesses[0], sizing_guesses[1] # M1/M2 (First Stage PMOS)
        gm_id_L, L_L = sizing_guesses[2], sizing_guesses[3] # M3/M4 (Bottom Sinks NMOS)
        gm_id_2, L_2 = sizing_guesses[4], sizing_guesses[5] # M9    (Second Stage PMOS)
        gm_id_3, L_3 = sizing_guesses[6], sizing_guesses[7] # M11   (Third Stage PMOS)
        gm_id_b, L_b = sizing_guesses[8], sizing_guesses[9] # M5/M6 (RAFFC Cascodes NMOS)
        
        vds_guess = self.VDD / 2.0 
        
        # --- 2. QUERY THE ANN FOR ALL 6 TARGETS ---
        id_w_1, gds_w_1, vgs_1, vdsat_1, cgg_w_1, cdd_w_1 = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=False)
        id_w_L, gds_w_L, vgs_L, vdsat_L, cgg_w_L, cdd_w_L = self.get_transistor_params(gm_id_L, L_L, vds_guess, is_nmos=True)
        id_w_2, gds_w_2, vgs_2, vdsat_2, cgg_w_2, cdd_w_2 = self.get_transistor_params(gm_id_2, L_2, vds_guess, is_nmos=False)
        id_w_3, gds_w_3, vgs_3, vdsat_3, cgg_w_3, cdd_w_3 = self.get_transistor_params(gm_id_3, L_3, vds_guess, is_nmos=False)
        id_w_b, gds_w_b, vgs_b, vdsat_b, cgg_w_b, cdd_w_b = self.get_transistor_params(gm_id_b, L_b, vds_guess, is_nmos=True)
        
        # Extract branch bias currents from optimizer
        Id_1 = sizing_guesses[10]
        Id_2 = sizing_guesses[11]
        Id_3 = sizing_guesses[12]
        Id_b = sizing_guesses[13]

        # Prevent division by zero or negative extrapolations
        id_w_1 = max(id_w_1, 1e-6)
        id_w_L = max(id_w_L, 1e-6)
        id_w_2 = max(id_w_2, 1e-6)
        id_w_3 = max(id_w_3, 1e-6)
        id_w_b = max(id_w_b, 1e-6)

        # Compute Physical Widths cleanly
        W_1 = Id_1 / id_w_1
        W_2 = Id_2 / id_w_2
        W_3 = Id_3 / id_w_3
        W_b = Id_b / id_w_b
        W_L = (Id_1 + Id_b) / id_w_L  # Sinks pull sum of diff pair and cascode branch

        # Calculate actual transconductances (gm = gm_id * Id)
        gm1 = gm_id_1 * Id_1
        gm2 = gm_id_2 * Id_2
        gm3 = gm_id_3 * Id_3
        gmb = gm_id_b * Id_b
        
        # CRITICAL UPDATE: Compute exact output resistances from normalized conductance
        ro1 = 1.0 / max(W_1 * gds_w_1, 1e-12)
        ro2 = 1.0 / max(W_2 * gds_w_2, 1e-12)
        ro3 = 1.0 / max(W_3 * gds_w_3, 1e-12)
        
        # --- 3. KVL HEADROOM TRACING & SATURATION CHECK ---
        VCM, VSS = 0.5, 0.0
        V_tail = VCM + abs(vgs_1)
        V_FS   = VSS + abs(vdsat_L) + 0.05
        V_out1 = self.VDD - abs(vgs_2)

        # Check available VDS against required VDSAT + 50mV margin
        margins = [
            (V_tail - V_FS) - abs(vdsat_1),  # M1/M2 Headroom
            (V_FS - VSS)    - abs(vdsat_L),  # M3/M4 Headroom
            (V_out1 - V_FS) - abs(vdsat_b)   # M5/M6 Headroom
        ]
        
        sat_penalty = 0.0
        for m in margins:
            if m < 0.05:
                sat_penalty += (0.05 - m)**2 * 1e6  # Heavy quadratic penalty for triode
        
        # --- 4. APPLY RAFFC ANALYTICAL EQUATIONS ---
        Av_linear = (gm1 * ro1) * (gm2 * ro2) * (gm3 * ro3)
        if Av_linear <= 0 or np.isnan(Av_linear):
            return 0, 0, -100, 1e6, 0
        DC_Gain_dB = 20 * np.log10(Av_linear)
        
        GBW_hz = gm1 / (2 * np.pi * self.CC1)
        
        # Phase Margin with Parasitic Loading from Stage 2 Gate (Cgg_W)
        ratio = gmb / max(gm1, 1e-9)
        ideal_pm_rad = np.arctan((ratio**3) / (ratio**2 + 2))
        ideal_pm_deg = np.degrees(ideal_pm_rad)
        
        # Compute parasitic pole delay loaded by M9 gate capacitance
        C_load_gate = W_2 * cgg_w_2
        omega_p_gate = gmb / max(C_load_gate, 1e-15)
        parasitic_delay_deg = np.degrees(np.arctan((2 * np.pi * GBW_hz) / omega_p_gate))
        
        PM_deg = ideal_pm_deg - parasitic_delay_deg
        
        # Stability and Extrapolation Barriers
        if gm1 >= gmb or id_w_1 <= 1e-4 or id_w_2 <= 1e-4 or id_w_3 <= 1e-4 or id_w_b <= 1e-4 or id_w_L <= 1e-4:
            return 0, 0, -100, 1e6, 0 
            
        CC2_req = (2 * gm3 * (self.CC1**2)) / (gmb * self.CL)
        
        # Compute Power and integrate the saturation penalty natively
        total_current = (Id_1 * 2) + Id_2 + (Id_3 * 2) + (Id_b * 2) 
        Power = (self.VDD * total_current) + sat_penalty

        if not np.isfinite(DC_Gain_dB) or not np.isfinite(GBW_hz) or not np.isfinite(PM_deg):
            return 0, 0, -100, 1e6, 0
        
        return DC_Gain_dB, GBW_hz, PM_deg, Power, CC2_req
    
    def calculate_physical_dimensions(self, final_optimal_guesses, bias_currents):
        """Prints the clean blueprint after optimization finishes."""
        gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b = final_optimal_guesses
        Id_1, Id_2, Id_3, Id_b = bias_currents
        vds_guess = self.VDD / 2.0 

        id_w_1, _, vgs_1, _, _, _ = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=False)
        id_w_L, _, vgs_L, _, _, _ = self.get_transistor_params(gm_id_L, L_L, vds_guess, is_nmos=True)
        id_w_2, _, vgs_2, _, _, _ = self.get_transistor_params(gm_id_2, L_2, vds_guess, is_nmos=False)
        id_w_3, _, vgs_3, _, _, _ = self.get_transistor_params(gm_id_3, L_3, vds_guess, is_nmos=False)
        id_w_b, _, vgs_b, _, _, _ = self.get_transistor_params(gm_id_b, L_b, vds_guess, is_nmos=True)

        W_1 = Id_1 / id_w_1
        W_2 = Id_2 / id_w_2
        W_3 = Id_3 / id_w_3
        W_b = Id_b / id_w_b
        W_L = (Id_1 + Id_b) / id_w_L  

        gm1 = gm_id_1 * Id_1
        gm3 = gm_id_3 * Id_3
        gmb = gm_id_b * Id_b

        # Biasing Network
        bias_gm_id, bias_L = 10.0, 400e-9
        id_w_pmos_bias, _, vgs_pmos_bias, _, _, _ = self.get_transistor_params(bias_gm_id, bias_L, vds_guess, is_nmos=False)
        id_w_nmos_bias, _, vgs_nmos_bias, _, _, _ = self.get_transistor_params(bias_gm_id, bias_L, vds_guess, is_nmos=True)

        W_M0 = (Id_1 * 2) / id_w_pmos_bias
        W_M7 = W_M8 = Id_b / id_w_pmos_bias
        W_M10 = Id_2 / id_w_nmos_bias
        W_M12 = W_M14 = Id_3 / id_w_nmos_bias
        # Ensure the extracted VGS is physically reasonable for a 1.0V supply
        # A typical moderate-inversion PMOS VGS in gpdk45 should sit between 0.4V and 0.55V
        vgs_pmos_clean = min(abs(vgs_pmos_bias), 0.55)
        # Bias Nodes
        VB1 = self.VDD - vgs_pmos_clean 
        VB2 = abs(vgs_nmos_bias)
        vds_sat_M3 = 2.0 / gm_id_L
        VB3 = abs(vgs_nmos_bias) + abs(vds_sat_M3)

        CC1_calc = gm1 / (2 * np.pi * 5e6)
        CC2_calc = (2 * gm3 * (CC1_calc**2)) / (gmb * self.CL)

        print("\n=== CORE AMPLIFIER TRANSISTORS ===")
        print(f"M1/M2 (Diff Pair) -> W: {W_1*1e6:.2f}um, L: {L_1*1e9:.0f}nm")
        print(f"M3/M4 (CM Load)   -> W: {W_L*1e6:.2f}um, L: {L_L*1e9:.0f}nm")
        print(f"M9    (2nd Stage) -> W: {W_2*1e6:.2f}um, L: {L_2*1e9:.0f}nm")
        print(f"M11/13(3rd Stage) -> W: {W_3*1e6:.2f}um, L: {L_3*1e9:.0f}nm")
        print(f"M5/M6 (RAFFC)     -> W: {W_b*1e6:.2f}um, L: {L_b*1e9:.0f}nm")

        print("\n=== BIASING NETWORK TRANSISTORS (gm/Id=10, L=400nm) ===")
        print(f"M0 (Tail Current)      -> W: {W_M0*1e6:.2f}um, L: 400nm")
        print(f"M7/M8 (Current Mirror) -> W: {W_M7*1e6:.2f}um, L: 400nm")
        print(f"M10   (2nd Stage Sink) -> W: {W_M10*1e6:.2f}um, L: 400nm")
        print(f"M12/14(3rd Stage Sinks)-> W: {W_M12*1e6:.2f}um, L: 400nm")
        
        print("\n=== CADENCE DC VOLTAGE SOURCES ===")
        print(f"VB1: {VB1:.3f} V")
        print(f"VB2: {VB2:.3f} V")
        print(f"VB3: {VB3:.3f} V")
        print("==================================\n")

        print("=== PASSIVE COMPONENTS (Capacitors) ===")
        print(f"Load Capacitor (CL) -> {self.CL*1e12:.2f} pF")
        print(f"Primary Comp (CC1)  -> {CC1_calc*1e12:.2f} pF")
        print(f"Secondary Comp(CC2) -> {CC2_calc*1e15:.2f} fF")
        print("==================================\n")
        
        return W_1, W_2, W_3, W_b

if __name__ == "__main__":
    testCircuit = RAFFC_OpAmp('nmos_surrogate_model.pth', 'pmos_surrogate_model.pth', 'scaler_X_nmos.pkl', 'scaler_y_nmos.pkl', 'scaler_X_pmos.pkl', 'scaler_y_pmos.pkl')
    preds = testCircuit.get_transistor_params(gm_id=2.0, L=45e-9, vds=1.0, is_nmos=True)
    print("Completed!")
    print(f"Id/W: {preds[0]:.2f} A/m, gds/W: {preds[1]:.2f} S/m, VGS: {preds[2]:.3f} V")