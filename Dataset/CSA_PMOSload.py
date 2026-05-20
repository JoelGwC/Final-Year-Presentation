import numpy as np
import torch
import joblib
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from model import SurrogateModel

class CommonSourceOptimizer(Problem):
    def __init__(self, nmos_model, pmos_model, scalers, specs):
        # 3 Variables: [gmid_1, L1, L2]
        # 2 Objectives: [Minimize -Gain (Maximize Gain), Minimize Id (Power)]
        # 2 Constraints: [fT1 >= 10GHz, W1 > 0 (valid bandwidth limit)]
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=5, 
                         xl=np.array([5.0, 45e-9, 45e-9]), 
                         xu=np.array([15.0, 200e-9, 300e-9]))
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.scalers = scalers
        self.specs = specs

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        
        # 1. Extract Variables
        gmid1 = X[:, 0]
        L1 = X[:, 1]
        L2 = X[:, 2]
        
        # Fixed biases
        vds1 = np.full(pop_size, self.specs['Vout'])
        vds2 = np.full(pop_size, self.specs['VDD'] - self.specs['Vout'])
        gmid2 = np.full(pop_size, self.specs['gmid2'])

        # 2. Prepare Inputs and Predict using PyTorch
        # NMOS inputs: [gmid, L, VDS]
        X_nmos = np.column_stack((gmid1, L1, vds1))
        X_nmos_scaled = self.scalers['X_nmos'].transform(X_nmos)
        
        # PMOS inputs: [gmid, L, VDS]
        X_pmos = np.column_stack((gmid2, L2, vds2))
        X_pmos_scaled = self.scalers['X_pmos'].transform(X_pmos)

        with torch.no_grad():
            Y_nmos_scaled = self.nmos_model(torch.tensor(X_nmos_scaled, dtype=torch.float32)).numpy()
            Y_pmos_scaled = self.pmos_model(torch.tensor(X_pmos_scaled, dtype=torch.float32)).numpy()

        Y_nmos = self.scalers['y_nmos'].inverse_transform(Y_nmos_scaled)
        Y_pmos = self.scalers['y_pmos'].inverse_transform(Y_pmos_scaled)

        # Map predictions to physical parameters
        # Output indices based on main.py: 0: Id_W, 1: gds_W, 2: VGS, 3: VDSAT, 4: Cgg_W, 5: Cdd_W
        id_w1 = np.maximum(Y_nmos[:, 0], 1e-12)
        gds_w1 = np.maximum(Y_nmos[:, 1], 1e-12)
        cgg_w1 = np.maximum(Y_nmos[:, 4], 1e-18)
        cdd_w1 = np.maximum(Y_nmos[:, 5], 1e-18)

        # Assuming you applied the absolute value fix for the PMOS from the previous step!
        id_w2 = np.maximum(np.abs(Y_pmos[:, 0]), 1e-12)
        gds_w2 = np.maximum(np.abs(Y_pmos[:, 1]), 1e-12)
        cdd_w2 = np.maximum(np.abs(Y_pmos[:, 5]), 1e-18)
        # 3. Calculate Derived Metrics
        # Transit Frequency of M1: fT = gm / (2 * pi * Cgg)
        gm_w1 = gmid1 * id_w1
        ft1 = gm_w1 / (2 * np.pi * cgg_w1)

        # intrinsic gains
        gds_id1 = gds_w1 / id_w1
        gds_id2 = gds_w2 / id_w2
        gain = gmid1 / np.maximum((gds_id1 + gds_id2), 1e-12)

        # Calculate Widths (W1 and W2)
        fu = self.specs['fu']
        Cl = self.specs['Cl']
        
        # The equation for the denominator of W1
        term_cap = 2 * np.pi * fu * (cdd_w1 + (id_w1 / id_w2) * cdd_w2)
        w1_denominator = gm_w1 - term_cap
        
        # Calculate W1, safely masking invalid negative denominators to a tiny positive number to avoid div-by-zero crashes
        safe_denom = np.where(w1_denominator > 0, w1_denominator, 1e-12)
        W1 = (2 * np.pi * fu * Cl) / safe_denom
        W2 = W1 * (id_w1 / id_w2)
        
        # Calculate Total Current
        Id = W1 * id_w1

        # 4. Assign Objectives and Constraints
        # Objectives (Minimize)
        f1 = -gain  # We want to maximize gain, so we minimize -gain
        f2 = Id     # Minimize Power consumption

        # Constraints (Must be <= 0)
        # Constraint 1: ft >= 10GHz -> 10GHz - ft <= 0
        g1 = self.specs['ft_target'] - ft1
        
        # Constraint 2: The device must be able to drive its own parasitic capacitance
        # w1_denominator must be > 0. Formulated as: 0.0 - w1_denominator <= 0
        g2 = -w1_denominator
        max_width = 500e-6 
        g3 = np.maximum(W1 - max_width, W2 - max_width)
        
        # Constraint 4: Total Current Id must be less than a strict power budget (e.g., 3 mA)
        max_current = 3e-3
        g4 = Id - max_current
        g5 = 15 - gain

        # Update the output dictionary
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5])


if __name__ == "__main__":
    print("Loading pre-trained models and scalers...")
    
    # 1. Load Scalers
    scalers = {
        'X_nmos': joblib.load('scaler_X_nmos.pkl'),
        'y_nmos': joblib.load('scaler_y_nmos.pkl'),
        'X_pmos': joblib.load('scaler_X_pmos.pkl'),
        'y_pmos': joblib.load('scaler_y_pmos.pkl')
    }
    
    # 2. Load PyTorch Models
    nmos_model = SurrogateModel()
    nmos_model.load_state_dict(torch.load('nmos_surrogate_model.pth', weights_only=True))
    nmos_model.eval()
    
    pmos_model = SurrogateModel()
    pmos_model.load_state_dict(torch.load('pmos_surrogate_model.pth', weights_only=True))
    pmos_model.eval()

    # 3. Define Design Specifications
    specs = {
        'fu': 1e9,          # 1 GHz
        'ft_target': 10e9,  # 10 GHz
        'Cl': 1e-12,        # 1 pF
        'VDD': 1.0,         # 1 V
        'Vout': 0.5,        # 0.5 V
        'gmid2': 10.0       # PMOS gm/Id
    }

    # 4. Initialize the Problem and Optimizer
    problem = CommonSourceOptimizer(nmos_model, pmos_model, scalers, specs)
    
    algorithm = NSGA2(pop_size=200)
    
    print("Running NSGA-II Optimization...")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=1,
                   verbose=True)

    # 5. Visualize the Pareto Front
    print("\nOptimization Complete. Generating Pareto Front.")
    plot = Scatter(title="Gain vs Current Trade-off (Pareto Front)")
    
    # Invert the objective 1 back to positive Gain for plotting
    f1_gain = -res.F[:, 0]
    f2_id = res.F[:, 1] * 1e6  # Convert to uA
    
    plot.add(np.column_stack([f1_gain, f2_id]))
    plot.show()

    # --- Post-Processing: Extracting Geometries ---
    print("\n--- Extracting Physical Dimensions ---")
    
    # 1. Choose a design point. 
    # Let's find a balanced design (e.g., a point with reasonable gain and current)
    # Sorting by gain to pick something in the middle of the Pareto front
    sorted_indices = np.argsort(f1_gain)
    chosen_index = sorted_indices[len(sorted_indices) // 2] # Pick the median design
    
    best_X = res.X[chosen_index]
    best_F = res.F[chosen_index]
    
    gmid1_opt = best_X[0]
    L1_opt = best_X[1]
    L2_opt = best_X[2]
    
    print(f"Selected Design - Predicted Gain: {-best_F[0]:.2f} V/V, Current: {best_F[1]*1e6:.2f} uA")
    print(f"Optimal L1 (NMOS): {L1_opt * 1e9:.2f} nm")
    print(f"Optimal L2 (PMOS): {L2_opt * 1e9:.2f} nm")
    print(f"Optimal gm/Id M1:  {gmid1_opt:.2f} S/A")

    # 2. Re-run the ANN for this specific point to get the Widths
    X_nmos_eval = np.array([[gmid1_opt, L1_opt, specs['Vout']]])
    X_pmos_eval = np.array([[specs['gmid2'], L2_opt, specs['VDD'] - specs['Vout']]])
    
    X_n_scaled = scalers['X_nmos'].transform(X_nmos_eval)
    X_p_scaled = scalers['X_pmos'].transform(X_pmos_eval)
    
    with torch.no_grad():
        Y_n_scaled = nmos_model(torch.tensor(X_n_scaled, dtype=torch.float32)).numpy()
        Y_p_scaled = pmos_model(torch.tensor(X_p_scaled, dtype=torch.float32)).numpy()
        
    Y_n = scalers['y_nmos'].inverse_transform(Y_n_scaled)[0]
    Y_p = scalers['y_pmos'].inverse_transform(Y_p_scaled)[0]
    
    id_w1_opt = Y_n[0]
    Vgs1_opt = Y_n[2]
    cdd_w1_opt = Y_n[5]
    id_w2_opt = Y_p[0]
    Vgs2_opt = Y_p[2]
    cdd_w2_opt = Y_p[5]
    
    # 3. Recalculate W1 and W2 using your algebraic derivation
    gm_w1_opt = gmid1_opt * id_w1_opt
    fu = specs['fu']
    Cl = specs['Cl']
    
    term_cap = 2 * np.pi * fu * (cdd_w1_opt + (id_w1_opt / id_w2_opt) * cdd_w2_opt)
    w1_denominator = gm_w1_opt - term_cap
    
    if w1_denominator > 0:
        W1_opt = (2 * np.pi * fu * Cl) / w1_denominator
        W2_opt = W1_opt * (id_w1_opt / id_w2_opt)
        
        print(f"Optimal W1 (NMOS): {W1_opt * 1e6:.2f} um")
        print(f"Optimal W2 (PMOS): {W2_opt * 1e6:.2f} um")
        print(f"Optimal VGS1 (NMOS): {Vgs1_opt:.2f} V")
        print(f"Optimal VGS2 (PMOS): {abs(Vgs2_opt):.2f} V")
    else:
        print("Warning: This specific design point cannot drive the required capacitance.")