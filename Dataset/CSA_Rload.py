import numpy as np
import torch
import joblib
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from model import SurrogateModel

class CommonSourceOptimizer(Problem):
    def __init__(self, nmos_model, scalers, specs):
        # 2 Variables: [gmid_1, L1]
        # 2 Objectives: [Minimize -Gain (Maximize Gain), Minimize Id (Power)]
        # 4 Constraints: [fT1 >= 10GHz, can drive load, W1 < 500um, Id < 3mA]
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=4, 
                         xl=np.array([5.0, 45e-9]), 
                         xu=np.array([15.0, 200e-9]))
        
        self.nmos_model = nmos_model
        self.scalers = scalers
        self.specs = specs

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        
        # 1. Extract Variables
        gmid1 = X[:, 0]
        L1 = X[:, 1]
          
        # Fixed bias
        vds1 = np.full(pop_size, self.specs['Vout'])    

        # 2. Prepare Inputs and Predict using PyTorch
        X_nmos = np.column_stack((gmid1, L1, vds1))
        X_nmos_scaled = self.scalers['X_nmos'].transform(X_nmos)
        
        with torch.no_grad():
            Y_nmos_scaled = self.nmos_model(torch.tensor(X_nmos_scaled, dtype=torch.float32)).numpy()
            
        Y_nmos = self.scalers['y_nmos'].inverse_transform(Y_nmos_scaled)
        
        # Map predictions and CLAMP to prevent divide-by-zero
        id_w1 = np.maximum(Y_nmos[:, 0], 1e-12)
        gds_w1 = np.maximum(Y_nmos[:, 1], 1e-12)
        cgg_w1 = np.maximum(Y_nmos[:, 4], 1e-18)
        cdd_w1 = np.maximum(Y_nmos[:, 5], 1e-18)

        # Transit Frequency of M1
        gm_w1 = gmid1 * id_w1
        ft1 = gm_w1 / (2 * np.pi * cgg_w1)

        # 3. Analytically calculate Id, W1, and R to guarantee 1 GHz bandwidth
        fu = self.specs['fu']
        Cl = self.specs['Cl']
        VDD = self.specs['VDD']
        Vout = self.specs['Vout']
        VR = VDD - Vout # Voltage across the resistor
        
        # Denominator of the Id equation
        id_denom = gmid1 - 2 * np.pi * fu * (cdd_w1 / id_w1)
        safe_denom = np.where(id_denom > 0, id_denom, 1e-12)
        
        # Calculate Physical Values
        Id = (2 * np.pi * fu * Cl) / safe_denom
        W1 = Id / id_w1
        R = VR / Id
        
        # Calculate Actual Loaded Gain: Av = gm1 * (R || ro1)
        gm1 = gmid1 * Id
        gds1 = W1 * gds_w1
        gain = gm1 / ((1.0 / R) + gds1)

        # 4. Assign Objectives and Constraints
        f1 = -gain  # Maximize gain
        f2 = Id     # Minimize Power consumption

        # Constraints (Must be <= 0)
        g1 = self.specs['ft_target'] - ft1     # fT >= 10 GHz
        g2 = -id_denom                         # Must be able to drive its own parasitics
        g3 = W1 - 500e-6                       # W1 <= 500 um
        g4 = Id - 3e-3                         # Id <= 3 mA
        # Removed the Gain >= 15 constraint because it is physically impossible with R load

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])


if __name__ == "__main__":
    print("Loading pre-trained models and scalers...")
    
    scalers = {
        'X_nmos': joblib.load('scaler_X_nmos.pkl'),
        'y_nmos': joblib.load('scaler_y_nmos.pkl')
    }
    
    nmos_model = SurrogateModel()
    nmos_model.load_state_dict(torch.load('nmos_surrogate_model.pth', weights_only=True))
    nmos_model.eval()

    specs = {
        'fu': 1e9,          # 1 GHz
        'ft_target': 10e9,  # 10 GHz
        'Cl': 1e-12,        # 1 pF
        'VDD': 1.0,         # 1 V
        'Vout': 0.5,        # 0.5 V
    }

    problem = CommonSourceOptimizer(nmos_model, scalers, specs)
    algorithm = NSGA2(pop_size=200)
    
    print("Running NSGA-II Optimization...")
    res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True)


    # --- Post-Processing: Extracting Geometries ---
    print("\n--- Extracting Physical Dimensions ---")
    f1_gain = -res.F[:, 0]
    f2_id = res.F[:, 1] * 1e6 
    sorted_indices = np.argsort(f1_gain)
    chosen_index = sorted_indices[len(sorted_indices) // 2] 
    
    best_X = res.X[chosen_index]
    best_F = res.F[chosen_index]
    
    gmid1_opt = best_X[0]
    L1_opt = best_X[1]
    
    print(f"Selected Design - Predicted Gain: {-best_F[0]:.2f} V/V, Current: {best_F[1]*1e6:.2f} uA")
    print(f"Optimal gm/Id M1: {gmid1_opt:.2f} S/A")

    # Re-run the ANN for this specific point to get parameters
    X_nmos_eval = np.array([[gmid1_opt, L1_opt, specs['Vout']]])
    X_n_scaled = scalers['X_nmos'].transform(X_nmos_eval)
    
    with torch.no_grad():
        Y_n_scaled = nmos_model(torch.tensor(X_n_scaled, dtype=torch.float32)).numpy()
        
    Y_n = scalers['y_nmos'].inverse_transform(Y_n_scaled)[0]
    
    id_w1_opt = Y_n[0]
    Vgs1_opt = Y_n[2]
    cdd_w1_opt = Y_n[5]
    
    # Recalculate physical dimensions
    id_denom_opt = gmid1_opt - 2 * np.pi * specs['fu'] * (cdd_w1_opt / id_w1_opt)
    
    if id_denom_opt > 0:
        Id_opt = (2 * np.pi * specs['fu'] * specs['Cl']) / id_denom_opt
        W1_opt = Id_opt / id_w1_opt
        R_opt = (specs['VDD'] - specs['Vout']) / Id_opt
        
        print(f"Optimal L1 (NMOS): {L1_opt * 1e9:.2f} nm")
        print(f"Optimal W1 (NMOS): {W1_opt * 1e6:.2f} um")
        print(f"Optimal ID1 (NMOS): {Id_opt * 1e6:.2f} um")
        print(f"Optimal Load Resistor (R): {R_opt / 1000:.2f} kOhm")
        print(f"Optimal VGS1 (NMOS): {Vgs1_opt:.3f} V")
    else:
        print("Warning: This specific design point cannot drive the required capacitance.")


    print("\nOptimization Complete. Generating Pareto Front.")
    plot = Scatter(title="Gain vs Current Trade-off (Pareto Front)")
    plot.add(np.column_stack([f1_gain, f2_id]))
    plot.show()