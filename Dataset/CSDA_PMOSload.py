import numpy as np
import torch
import joblib
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from model import SurrogateModel
from getparams import get_transistor_params

def resolve_operating_point(gmid1, L1, L2, specs):
    """
    Iteratively resolves VDS and VGS for a diff-pair with a PMOS mirror load.
    Returns: (NMOS_tuple, PMOS_tuple)
    """
    # 1. Setup initial dummy guesses (multiplying by 0 makes it safe for both arrays and scalars)
    gmid2 = gmid1 * 0 + specs['gmid2']
    vds_guess = gmid1 * 0 + 0.5
    
    # 2. Resolve the PMOS Mirror Load (Diode-Connected Side)
    # Guess VDS=0.5 to get an initial VGS2
    _, _, vgs2_init, _, _, _ = get_transistor_params(gmid2, L2, vds_guess, is_nmos=False)
    # Re-evaluate exactly at VDS2 = VGS2
    id_w2, gds_w2, vgs2, vdsat2, cgg_w2, cdd_w2 = get_transistor_params(gmid2, L2, vgs2_init, is_nmos=False)
    
    # 3. Resolve the NMOS Input Pair
    # Guess VDS=0.5 to get an initial VGS1
    _, _, vgs1_init, _, _, _ = get_transistor_params(gmid1, L1, vds_guess, is_nmos=True)
    
    # Calculate exact voltages using Kirchhoff's Voltage Law
    Vs = specs['ICM'] - vgs1_init
    Vout = specs['VDD'] - vgs2
    vds1_real = Vout - Vs
    
    # Re-evaluate NMOS at its true floating VDS
    id_w1, gds_w1, vgs1, vdsat1, cgg_w1, cdd_w1 = get_transistor_params(gmid1, L1, vds1_real, is_nmos=True)
    
    # Package the results
    nmos_params = (id_w1, gds_w1, vgs1, cgg_w1, cdd_w1, vds1_real)
    pmos_params = (id_w2, gds_w2, vgs2, cgg_w2, cdd_w2, vgs2_init) # PMOS VDS is VGS
    
    return nmos_params, pmos_params
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
        vds1 = np.full(pop_size, 0.5)
        vds2 = np.full(pop_size, 0.5)
        gmid2 = np.full(pop_size, self.specs['gmid2'])

        # 2. Prepare Inputs and Predict using PyTorch
        # Map predictions to physical parameters
        # Output indices based on main.py: 0: Id_W, 1: gds_W, 2: VGS, 3: VDSAT, 4: Cgg_W, 5: Cdd_W
        nmos_params, pmos_params = resolve_operating_point(gmid1, L1, L2, self.specs)
        id_w1, gds_w1, vgs1, cgg_w1, cdd_w1, vds1 = nmos_params
        id_w2, gds_w2, vgs2, cgg_w2, cdd_w2, vds2 = pmos_params

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
        Vs1 = self.specs['ICM'] - vgs1
        
        # We demand at least 150mV (0.15V) of headroom for the tail current source.
        # Required math: Vs1 >= 0.15  -->  0.15 - Vs1 <= 0
        min_tail_headroom = 0.15
        g6 = min_tail_headroom - Vs1

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
        'gmid2': 10.0,       # PMOS gm/Id
        'ICM': 0.7
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
    #how to pass the predicted vds from _evaluate function? 
    n_params, p_params = resolve_operating_point(gmid1_opt, L1_opt, L2_opt, specs)
    id_wn_opt, gds_wn_opt, vgsn_opt, cgg_wn_opt, cdd_wn_opt, vdsn_opt = n_params
    id_wp_opt, gds_wp_opt, vgsp_opt, cgg_wp_opt, cdd_wp_opt, vdsp_opt = p_params
    
    # 3. Recalculate W1 and W2 using your algebraic derivation
    gm_w1_opt = gmid1_opt * id_wn_opt
    fu = specs['fu']
    Cl = specs['Cl']
    
    term_cap = 2 * np.pi * fu * (cdd_wn_opt + (id_wn_opt / id_wp_opt) * cdd_wp_opt)
    w1_denominator = gm_w1_opt - term_cap
    
    if w1_denominator > 0:
        W1_opt = (2 * np.pi * fu * Cl) / w1_denominator
        W2_opt = W1_opt * (id_wn_opt / id_wp_opt)
        
        print(f"Optimal W1 (NMOS): {W1_opt * 1e6:.2f} um")
        print(f"Optimal W2 (PMOS): {W2_opt * 1e6:.2f} um")
        print(f"Optimal VGS1 (NMOS): {vgsn_opt:.2f} V")
        print(f"Optimal VGS2 (PMOS): {abs(vgsp_opt):.2f} V")
    else:
        print("Warning: This specific design point cannot drive the required capacitance.")
    
    
    plot.add(np.column_stack([f1_gain, f2_id]))
    plot.show()
    
    
    # 4. Sizing the Tail Current Source
    Vs1 = specs['ICM'] - vgsn_opt
    
    # We want VDSAT to be at least 50mV less than the available headroom (Vs1)
    # to guarantee it stays deeply in saturation.
    # target_vdsat = Vs1 - 0.050 
    
    # if target_vdsat <= 0:
    #     print("CRITICAL ERROR: No voltage headroom left for the tail current source!")
    #     print(f"ICM is {specs['ICM']}V, but input VGS is {vgsn_opt:.2f}V.")
    # else:
    #     # Loop through gm/Id from 5 to 25 to find the lowest one that satisfies our headroom
    #     optimal_gmid_tail = None
    #     for g_test in np.arange(5.0, 25.0, 0.5):
    #         _, _, _, vdsat_test, _, _ = get_transistor_params(g_test, 400e-9, Vs1, is_nmos=True)
            
    #         if vdsat_test < target_vdsat:
    #             optimal_gmid_tail = g_test
    #             break # We found the best (lowest possible) gm/Id that fits!
                
    #     if optimal_gmid_tail is None:
    #          print("Warning: Could not find a gm/Id that keeps the tail in saturation.")
    #          optimal_gmid_tail = 25.0 # Fallback to safest headroom
             
        # Now extract the final parameters using the safe gm/Id
    id_wtail, gds_wtail, vgstail, vdsattail, cgg_wtail, cdd_wtail =get_transistor_params(
            15, 400e-9, Vs1, is_nmos=True)
            
    W_tail = (2 * best_F[1]) / id_wtail
        
    print("\n--- Tail Current Source ---")
    print(f"Available Headroom (Vs1): {Vs1 * 1000:.2f} mV")
    print(f"Tail VDSAT: {vdsattail * 1000:.2f} mV")
    print(f"Optimal gm/Id (Tail): {15:.2f} S/A")
    print(f"Optimal L_tail (NMOS): 400.00 nm")
    print(f"Optimal W_tail (NMOS): {W_tail * 1e6:.2f} um")
    print(f"Optimal VGS_tail (NMOS): {vgstail:.3f} V")
