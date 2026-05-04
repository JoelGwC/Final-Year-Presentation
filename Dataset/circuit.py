import torch
import numpy as np
import joblib # For loading your scalers from Phase 1
from model import SurrogateModel # Import your architecture

class RAFFC_OpAmp:
    def __init__(self, nmos_pth, pmos_pth, scaler_X_path, scaler_y_path):
        # 1. Load the Scalers
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # 2. Load the NMOS Brain
        self.nmos_model = SurrogateModel()
        self.nmos_model.load_state_dict(torch.load(nmos_pth))
        self.nmos_model.eval() # CRITICAL: Sets model to inference mode
        
        # 3. Load the PMOS Brain
        # self.pmos_model = SurrogateModel()
        # self.pmos_model.load_state_dict(torch.load(pmos_pth))
        # self.pmos_model.eval()
        
        # Define Circuit Constants
        self.VDD = 1.0
        self.CL = 5e-12 # 5pF Load Capacitor

    def get_transistor_params(self, gm_id, L, vds, is_nmos=True):
        """The Bridge: Converts guesses into physical device metrics using the ANN"""
        
        # Format input and scale it
        raw_inputs = np.array([[gm_id, L, vds]])
        scaled_inputs = self.scaler_X.transform(raw_inputs)
        tensor_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
        
        # Query the Neural Network (Ultra-fast)
        with torch.no_grad():
            if is_nmos:
                scaled_preds = self.nmos_model(tensor_inputs).numpy()
            else:
                # scaled_preds = self.pmos_model(tensor_inputs).numpy()
                pass
                
        # Un-scale the predictions back to real-world physical values
        real_preds = self.scaler_y.inverse_transform(scaled_preds)
        id_w, gm_gds = real_preds[0]
        
        return id_w, gm_gds

    def evaluate(self, sizing_guesses):
        """
        Calculates the Amplifier specs. 
        sizing_guesses is an array provided by the Genetic Algorithm.
        E.g., [gm_id_M1, L_M1, gm_id_M3, L_M3, ...]
        """
        
        # Example extracting variables for the Differential Pair (M1 & M2)
        gm_id_1 = sizing_guesses[0]
        L_1 = sizing_guesses[1]
        
        # Assume VDS is roughly VDD/2 for this example
        vds_guess = self.VDD / 2.0 
        
        # 1. Ask the AI what M1 looks like at this operating point
        id_w_1, gm_gds_1 = self.get_transistor_params(gm_id_1, L_1, vds_guess, is_nmos=True)
        
        # 2. Derive the explicit physical metrics
        # If we set a bias current (Id) of 10uA per branch:
        Id_1 = 10e-6
        gm_1 = gm_id_1 * Id_1
        gds_1 = gm_1 / gm_gds_1
        ro_1 = 1.0 / gds_1
        W_1 = Id_1 / id_w_1 # The AI just sized the transistor width for us!
        
        # Repeat for M3, M4, etc...
        
        # 3. Calculate Final Circuit Equations (Examples)
        # DC_Gain = gm_1 * (ro_1 || ro_3) * ...
        # GBW = gm_1 / (2 * np.pi * self.CL)
        # Power = self.VDD * Total_Current
        
        # return DC_Gain, GBW, Power
        pass