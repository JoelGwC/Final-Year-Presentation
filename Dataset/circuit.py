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
        # self.CL = 500e-12    # 500 pF Load Capacitance
        self.CC1 = 11e-12    # 11 pF Primary Compensation Capacitor

    def get_transistor_params(self, gm_id, L, vds, is_nmos=True):
        """Queries the ANN to get Id/W and gm/gds for a given bias state."""
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
                
        id_w, gm_gds, vgs = real_preds[0]
        
        return id_w, gm_gds, vgs

    def evaluate(self, sizing_guesses):
        """
        sizing_guesses structure: 
        [gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b, Id_1, Id_2, Id_3, Id_b]
        """
        # --- 1. EXTRACT OPTIMIZER GUESSES ---
        gm_id_1, L_1 = sizing_guesses[0], sizing_guesses[1] # M1/M2 (First Stage)
        gm_id_L, L_L = sizing_guesses[2], sizing_guesses[3] # M3/M4 Load
        gm_id_2, L_2 = sizing_guesses[4], sizing_guesses[5]
        gm_id_3, L_3 = sizing_guesses[6], sizing_guesses[7]
        gm_id_b, L_b = sizing_guesses[8], sizing_guesses[9]
        
        # Note: M13 (Feedforward) gm must match M11 (gmf = gm3) per the paper
        
        # Assume VDS is roughly VDD/2 for extraction purposes
        vds_guess = self.VDD / 2.0 
        
        # --- 2. QUERY THE ANN FOR PHYSICAL PARAMETERS ---
        # Note: In the actual circuit, map is_nmos correctly based on Fig 6.
        id_w_1, gm_gds_1, _ = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=False)
        id_w_L, gm_gds_L, _ = self.get_transistor_params(gm_id_L, L_L, vds_guess, is_nmos=True)
        id_w_2, gm_gds_2, _ = self.get_transistor_params(gm_id_2, L_2, vds_guess, is_nmos=False)
        id_w_3, gm_gds_3, _ = self.get_transistor_params(gm_id_3, L_3, vds_guess, is_nmos=False)
        id_w_b, gm_gds_b, _ = self.get_transistor_params(gm_id_b, L_b, vds_guess, is_nmos=True)
        

        gm_gds_1 = max(gm_gds_1, 1e-9)
        gm_gds_2 = max(gm_gds_2, 1e-9)
        gm_gds_3 = max(gm_gds_3, 1e-9)

        # Extract branch bias currents from optimizer
        Id_1 = sizing_guesses[10]
        Id_2 = sizing_guesses[11]
        Id_3 = sizing_guesses[12]
        Id_b = sizing_guesses[13]
        
        # Calculate actual transconductances (gm = gm_id * Id)
        gm1 = gm_id_1 * Id_1
        gm2 = gm_id_2 * Id_2
        gm3 = gm_id_3 * Id_3
        gmb = gm_id_b * Id_b
        
        # Calculate output resistances (ro = gm_gds / gm)
        ro1 = gm_gds_1 / max(gm1, 1e-9)
        ro2 = gm_gds_2 / max(gm2, 1e-9)
        ro3 = gm_gds_3 / max(gm3, 1e-9)
       
        
        # --- 3. APPLY RAFFC ANALYTICAL EQUATIONS (From Grasso et al.) ---
        
        # A. DC Open-Loop Gain (Magnitude)
        Av_linear = (gm1 * ro1) * (gm2 * ro2) * (gm3 * ro3)
        if Av_linear <= 0 or np.isnan(Av_linear):
            return 0, 0, -100, 1e6, 0
        DC_Gain_dB = 20 * np.log10(Av_linear)
        
        # B. Gain-Bandwidth Product (Hz)
        GBW_hz = gm1 / (2 * np.pi * self.CC1)
        
        # C. Phase Margin (Degrees)
        ratio = gmb / gm1
        phase_margin_rad = np.arctan((ratio**3) / (ratio**2 + 2))
        PM_deg = np.degrees(phase_margin_rad)
        
        # D. Asymptotic Stability Penalty
        # If gm1 >= gmb, the amplifier is unstable. We return a massive penalty.
        if gm1 >= gmb:
            return 0, 0, -100, 1e6, 0 # Gain=0, GBW=0, PM=-100 (Failed design)
            
        # Neural Network Hallucination Penalty
        # If the ANN predicts a negative or near-zero Id/W, it means the optimizer pushed 
        # the parameters into an invalid extrapolation region. Penalize it heavily.
        if id_w_1 <= 1e-4 or id_w_2 <= 1e-4 or id_w_3 <= 1e-4 or id_w_b <= 1e-4 or id_w_L <= 1e-4:
            return 0, 0, -100, 1e6, 0 
            
        # E. Predict Required CC2
        CC2_req = (2 * gm3 * (self.CC1**2)) / (gmb * self.CL)
        
        # Calculate Total Power
        total_current = (Id_1 * 2) + Id_2 + (Id_3*2) + (Id_b*2) # Approximated
        Power = self.VDD * total_current

        if not np.isfinite(DC_Gain_dB) or not np.isfinite(GBW_hz) or not np.isfinite(PM_deg):
            return 0, 0, -100, 1e6, 0
        
        return DC_Gain_dB, GBW_hz, PM_deg, Power, CC2_req
    
    def calculate_physical_dimensions(self, final_optimal_guesses, bias_currents):
        """Run this ONCE after the optimizer finishes to get Cadence widths."""
        gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b = final_optimal_guesses
        Id_1, Id_2, Id_3, Id_b = bias_currents
        vds_guess = self.VDD / 2.0 

    
        # Extract Id/W from the ANN
        id_w_1, _, vgs_1 = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=False)
        id_w_L, _, vgs_L = self.get_transistor_params(gm_id_L, L_L, vds_guess, is_nmos=True)
        id_w_2, _, vgs_2 = self.get_transistor_params(gm_id_2, L_2, vds_guess, is_nmos=False)
        id_w_3, _, vgs_3 = self.get_transistor_params(gm_id_3, L_3, vds_guess, is_nmos=False)
        id_w_b, _, vgs_b = self.get_transistor_params(gm_id_b, L_b, vds_guess, is_nmos=True)

        # Calculate final physical Widths (W = Id / (Id/W))
        W_1 = Id_1 / id_w_1
        W_2 = Id_2 / id_w_2
        W_3 = Id_3 / id_w_3
        W_b = Id_b / id_w_b
        W_L = (Id_1 + Id_b) / id_w_L  # Width for M3 and M4

        gm1 = gm_id_1 * Id_1
        gm3 = gm_id_3 * Id_3
        gmb = gm_id_b * Id_b

        # ==========================================
        # 4. BIASING NETWORK CALCULATION
        # ==========================================
        # Fixed state for current mirrors (Strong inversion, long channel)

        bias_gm_id = 10.0
        bias_L = 400e-9
        # Query the network for the biasing transistors
        id_w_pmos_bias, _, vgs_pmos_bias = self.get_transistor_params(bias_gm_id, bias_L, vds_guess, is_nmos=False)
        id_w_nmos_bias, _, vgs_nmos_bias = self.get_transistor_params(bias_gm_id, bias_L, vds_guess, is_nmos=True)

        # ------------------------------------------
        # Size the PMOS Current Sources (connected to VDD)
        # ------------------------------------------
    
        # Calculate widths for the bias transistors based on the current they must supply
        # Example: M0 supplies the tail current for both M1 and M2 (Id_1 * 2)
        W_M0 = (Id_1 * 2) / id_w_pmos_bias
        # M7/M8 provide bias currents for the folded branches (Assume Id_2 for this example)
        W_M7 = W_M8 = Id_b / id_w_pmos_bias
    
        # -- NMOS Cascode (Bias with gm/Id=10 to keep it matched) --
        W_M5 = W_b         # M5 simply passes Id_b
        L_M5 = L_b         # M5 simply passes Id_b

        # ------------------------------------------
        # Size the NMOS Current Sinks (connected to VSS)
        # ------------------------------------------
        # M10 provides the sink for the 2nd stage (M9)

        W_M10 = Id_2 / id_w_nmos_bias
        # M12/M14 provide the sinks for the 3rd stage and Feedforward (M11/M13)
        W_M12 = Id_3 / id_w_nmos_bias
        W_M14 = Id_3 / id_w_nmos_bias
        # ------------------------------------------
        # Bias Voltages (Adjusted for absolute values)
        # ------------------------------------------
        # VB1: Gate of PMOS current sources
        # The NMOS current mirror (M10) sets the gate voltage for M7/M8
        VB1 = self.VDD - abs(vgs_pmos_bias) 

        # VB2: Gate of NMOS current sources
        # VSS(0V) + Vgs(NMOS) = Vgs
        VB2 = abs(vgs_nmos_bias)

        # VB3: Gate of Cascode M5
        # Needs to be Vgs_nmos + Vds_sat of M3
        vds_sat_M3 = 2.0 / gm_id_L  # Approximation for saturation voltage
        VB3 = abs(vgs_nmos_bias) + abs(vds_sat_M3)

        # Calculate Bias Node Voltages (KVL)
        # PMOS Vgs is negative. E.g., if vgs_pmos is -0.6V, Gate is at 3.0V - 0.6V = 2.4V
        VB1 = self.VDD - abs(vgs_pmos_bias) 
        
        # NMOS sources are at 0V (VSS), so Gate is exactly Vgs
        VB2 = abs(vgs_nmos_bias)
        
        # VB3 is for a cascode (M5). It must provide enough voltage to keep M3 in saturation.
        # VDS,sat is roughly 2 / (gm/Id). So we add M3's saturation voltage to M5's VGS.
        vds_sat_M3 = 2.0 / gm_id_L
        VB3 = abs(vgs_nmos_bias) + abs(vds_sat_M3)

        CL = 2e-12 

        # 3. Calculate CC1 based on your 5MHz GBW target
        target_gbw_hz = 5e6
        CC1_calc = gm1 / (2 * np.pi * target_gbw_hz)

        # 4. Calculate CC2 using your exact Butterworth RAFFC equation
        CC2_calc = (2 * gm3 * (CC1_calc**2)) / (gmb * CL)

        # ==========================================
        # 5. PRINT THE FINAL BLUEPRINT
        # ==========================================
        print("\n=== CORE AMPLIFIER TRANSISTORS ===")
        print(f"M1/M2 (Diff Pair) -> W: {W_1*1e6:.2f}um, L: {L_1*1e9:.0f}nm")
        print(f"M3/M4 (CM Load)   -> W: {W_L*1e6:.2f}um, L: {L_L*1e9:.0f}nm")
        print(f"M9    (2nd Stage) -> W: {W_2*1e6:.2f}um, L: {L_2*1e9:.0f}nm")
        print(f"M11/13(3rd Stage) -> W: {W_3*1e6:.2f}um, L: {L_3*1e9:.0f}nm")
        print(f"M5/M6    (RAFFC)     -> W: {W_b*1e6:.2f}um, L: {L_b*1e9:.0f}nm")

        print("\n=== BIASING NETWORK TRANSISTORS (gm/Id=10, L=400nm) ===")
        print(f"M0 (Tail Current) -> W: {W_M0*1e6:.2f}um, L: 400nm")
        print(f"M7/M8 (Current Mirror) -> W: {W_M7*1e6:.2f}um, L: 400nm")
        print(f"M10 (2nd Stage Sink) -> W: {W_M10*1e6:.2f}um, L: 400nm")
        print(f"M12/M14 (3rd Stage Sinks) -> W: {W_M12*1e6:.2f}um, L: 400nm")
        
        print("\n=== CADENCE DC VOLTAGE SOURCES ===")
        print(f"VB1: {VB1:.3f} V")
        print(f"VB2: {VB2:.3f} V")
        print(f"VB3: {VB3:.3f} V")
        print("==================================\n")

        print("=== PASSIVE COMPONENTS (Capacitors) ===")
        print(f"Load Capacitor (CL) -> {CL*1e12:.2f} pF")
        print(f"Primary Comp (CC1)  -> {CC1_calc*1e12:.2f} pF")
        print(f"Secondary Comp(CC2) -> {CC2_calc*1e15:.2f} fF")
        print("==================================\n")

       

        # print(f"Final M1/M2 Dimensions (gm1) -> W: {W_1*1e6:.2f}um, L: {L_1*1e9:.0f}nm, VGS: {vgs_1:.3f}V")
        # print(f"Final M3/M4 Dimensions -> W: {W_L*1e6:.2f}um, L: {L_L*1e9:.0f}nm, VGS (VB2): {vgs_L:.3f}V")
        # print(f"Final M9 Dimensions (gm2) -> W: {W_2*1e6:.2f}um, L: {L_2*1e9:.0f}nm, VGS: {vgs_2:.3f}V")
        # print(f"Final M11/M13 Dimensions (gm3)-> W: {W_3*1e6:.2f}um, L: {L_3*1e9:.0f}nm, VGS: {vgs_3:.3f}V")
        # print(f"Final M6 Dimensions (gmb)-> W: {W_b*1e6:.2f}um, L: {L_b*1e9:.0f}nm, VGS: {vgs_b:.3f}V")
        # print(f"Estimated VB3 (Gate of M5/M6) -> Assuming M4 VDS={vds_guess:.2f}V, VB3 = VDS_M4 + VGS_M6 = {vds_guess:.2f} + {vgs_b:.3f} = {vds_guess + vgs_b:.3f}V")
        
        return W_1, W_2, W_3, W_b
    

if __name__ == "__main__":

    testCircuit = RAFFC_OpAmp('nmos_surrogate_model.pth', 'pmos_surrogate_model.pth', 'scaler_X_nmos.pkl', 'scaler_y_nmos.pkl', 'scaler_X_pmos.pkl', 'scaler_y_pmos.pkl')
    id_w, gm_gds, vgs = testCircuit.get_transistor_params(gm_id=2.0, L=45e-9, vds=1.0, is_nmos=True)
    print("Completed!")
    print(f"{id_w:.2f} uA/um, {gm_gds:.2f} S/S, {vgs:.3f} V")