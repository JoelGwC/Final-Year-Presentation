import torch
import numpy as np
import joblib
from model import SurrogateModel 

class RAFFC_OpAmp:
    def __init__(self, nmos_pth, pmos_pth, scaler_X_path, scaler_y_path):
        # Load scalers to normalize inputs/outputs for the ANN
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # Load the trained Surrogate Models (ANNs)
        self.nmos_model = SurrogateModel()
        self.nmos_model.load_state_dict(torch.load(nmos_pth))
        self.nmos_model.eval()
        
        # self.pmos_model = SurrogateModel()
        # self.pmos_model.load_state_dict(torch.load(pmos_pth))
        # self.pmos_model.eval()
        
        # Global Circuit Specifications (from the paper's test bench)
        self.VDD = 3.0       # Volts
        self.CL = 500e-12    # 500 pF Load Capacitance
        self.CC1 = 11e-12    # 11 pF Primary Compensation Capacitor

    def get_transistor_params(self, gm_id, L, vds, is_nmos=True):
        """Queries the ANN to get Id/W and gm/gds for a given bias state."""
        raw_inputs = np.array([[gm_id, L, vds]])
        scaled_inputs = self.scaler_X.transform(raw_inputs)
        tensor_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
        
        with torch.no_grad():
            if is_nmos:
                scaled_preds = self.nmos_model(tensor_inputs).numpy()
            else:
                # scaled_preds = self.pmos_model(tensor_inputs).numpy()
                pass
                
        real_preds = self.scaler_y.inverse_transform(scaled_preds)
        id_w, gm_gds = real_preds[0]
        return id_w, gm_gds

    def evaluate(self, sizing_guesses):
        """
        sizing_guesses structure: 
        [gm_id_1, L_1, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b]
        """
        # --- 1. EXTRACT OPTIMIZER GUESSES ---
        gm_id_1, L_1 = sizing_guesses[0], sizing_guesses[1] # M1/M2 (First Stage)
        gm_id_2, L_2 = sizing_guesses[2], sizing_guesses[3] # M9 (Second Stage)
        gm_id_3, L_3 = sizing_guesses[4], sizing_guesses[5] # M11 (Third Stage)
        gm_id_b, L_b = sizing_guesses[6], sizing_guesses[7] # M6 (Feedback Stage)
        
        # Note: M13 (Feedforward) gm must match M11 (gmf = gm3) per the paper
        
        # Assume VDS is roughly VDD/2 for extraction purposes
        vds_guess = self.VDD / 2.0 
        
        # --- 2. QUERY THE ANN FOR PHYSICAL PARAMETERS ---
        # Note: In the actual circuit, map is_nmos correctly based on Fig 6.
        id_w_1, gm_gds_1 = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=False)
        id_w_2, gm_gds_2 = self.get_transistor_params(gm_id_2, L_2, vds_guess, is_nmos=False)
        id_w_3, gm_gds_3 = self.get_transistor_params(gm_id_3, L_3, vds_guess, is_nmos=False)
        id_w_b, gm_gds_b = self.get_transistor_params(gm_id_b, L_b, vds_guess, is_nmos=True)
        
        # Set branch bias currents (Example values)
        Id_1 = 10e-6  # 10uA for first stage
        Id_2 = 20e-6  # 20uA for second stage
        Id_3 = 20e-6  # 20uA for third stage
        Id_b = 20e-6  # 20uA for feedback stage
        
        # Calculate actual transconductances (gm = gm_id * Id)
        gm1 = gm_id_1 * Id_1
        gm2 = gm_id_2 * Id_2
        gm3 = gm_id_3 * Id_3
        gmb = gm_id_b * Id_b
        
        # Calculate output resistances (ro = gm_gds / gm)
        ro1 = gm_gds_1 / gm1
        ro2 = gm_gds_2 / gm2
        ro3 = gm_gds_3 / gm3
        
        # --- 3. APPLY RAFFC ANALYTICAL EQUATIONS (From Grasso et al.) ---
        
        # A. DC Open-Loop Gain (Magnitude)
        Av_linear = (gm1 * ro1) * (gm2 * ro2) * (gm3 * ro3)
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
            return 0, 0, -100 # Gain=0, GBW=0, PM=-100 (Failed design)
            
        # E. Predict Required CC2
        CC2_req = (2 * gm3 * (self.CC1**2)) / (gmb * self.CL)
        
        # Calculate Total Power
        total_current = (Id_1 * 2) + Id_2 + Id_3 + Id_b # Approximated
        Power = self.VDD * total_current
        
        return DC_Gain_dB, GBW_hz, PM_deg, Power, CC2_req
    
    def calculate_physical_dimensions(self, final_optimal_guesses, bias_currents):
        """Run this ONCE after the optimizer finishes to get Cadence widths."""
        gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b = final_optimal_guesses
        Id_1, Id_2, Id_3, Id_b = bias_currents
        vds_guess = self.VDD / 2.0 

        # Extract Id/W from the ANN
        id_w_1, _ = self.get_transistor_params(gm_id_1, L_1, vds_guess)
        id_w_2, _ = self.get_transistor_params(gm_id_2, L_2, vds_guess)
        id_w_3, _ = self.get_transistor_params(gm_id_3, L_3, vds_guess)
        id_w_b, _ = self.get_transistor_params(gm_id_b, L_b, vds_guess)
        id_w_L, _ = self.get_transistor_params(gm_id_L, L_L, vds_guess, is_nmos=True)

        # Calculate final physical Widths (W = Id / (Id/W))
        W_1 = Id_1 / id_w_1
        W_2 = Id_2 / id_w_2
        W_3 = Id_3 / id_w_3
        W_b = Id_b / id_w_b
        W_L = Id_1 / id_w_L  # Width for M3 and M4 (ADDED!)

        print(f"Final M1 Dimensions -> W: {W_1*1e6:.2f}um, L: {L_1*1e9:.0f}nm")
        print(f"Final M3/M4 Dimensions -> W: {W_L*1e6:.2f}um, L: {L_L*1e9:.0f}nm") # ADDED!
        print(f"Final M9 Dimensions -> W: {W_2*1e6:.2f}um, L: {L_2*1e9:.0f}nm")
        print(f"Final M11 Dimensions-> W: {W_3*1e6:.2f}um, L: {L_3*1e9:.0f}nm")
        print(f"Final M6 Dimensions -> W: {W_b*1e6:.2f}um, L: {L_b*1e9:.0f}nm")
        
        return W_1, W_2, W_3, W_b
    

if __name__ == "__main__":

    testCircuit = RAFFC_OpAmp('nmos_surrogate_model.pth', 'pmos_surrogate_model.pth', 'scaler_X.pkl', 'scaler_y.pkl')
    id_w, gm_gds = testCircuit.get_transistor_params(gm_id=2.0, L=45e-9, vds=1.0, is_nmos=True)
    print("Completed!")
    print(f"{id_w:.2f} uA/um, {gm_gds:.2f} S/S")