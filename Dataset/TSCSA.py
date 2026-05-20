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
        # 4 Variables: [gmid_1, gmid_6, L1, L2, L6]
        # 2 Objectives: [Minimize -Gain (Maximize Gain), Minimize Id (Power)]
        # 2 Constraints: [fT1 >= 10GHz, W1 > 0 (valid bandwidth limit)]
        super().__init__(n_var=5, n_obj=2, n_ieq_constr=7, 
                         xl=np.array([5.0, 5.0, 45e-9, 45e-9, 45e-9]), 
                         xu=np.array([15.0, 15.0, 200e-9, 300e-9, 300e-9]))
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.scalers = scalers
        self.specs = specs

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        
        # 1. Extract Variables
        gmid1 = X[:, 0]
        gmid6 = X[:, 1]
        L1 = X[:, 2]
        L2 = X[:, 3]
        L6 = X[:, 4]
        
        # Fixed biases
        gmid2 = np.full(pop_size, self.specs['gmid2'])
        gmid5 = np.full(pop_size, self.specs['gmid5'])
        vds5 = np.full(pop_size, self.specs['VDD'] - self.specs['VOUT'])
        vds6 = np.full(pop_size, self.specs['VOUT'])
        L5 = L2

        # 2. Prepare Inputs and Predict using PyTorch
        # Map predictions to physical parameters
        # Output indices based on main.py: 0: Id_W, 1: gds_W, 2: VGS, 3: VDSAT, 4: Cgg_W, 5: Cdd_W
        nmos_params, pmos_params = resolve_operating_point(gmid1, L1, L2, self.specs)
        id_w1, gds_w1, vgs1, cgg_w1, cdd_w1, vds1 = nmos_params
        id_w2, gds_w2, vgs2, cgg_w2, cdd_w2, vds2 = pmos_params

        id_w5, gds_w5, vgs5, vdsat5, cgg_w5, cdd_w5 = get_transistor_params(gmid5, L5, vds5, is_nmos=False)
        id_w6, gds_w6, vgs6, vdsat6, cgg_w6, cdd_w6 = get_transistor_params(gmid6, L6, vds6, is_nmos=True)

        #Calculate Transistor Widths and Currents
        fu = self.specs['fu']
        Cl = self.specs['Cl']
        Cc = self.specs['Cc']

        #First differential stage
        gm1 = 2*np.pi*Cc*fu
        id1 = gm1/gmid1
        W1 = id1/id_w1
        W2 = id1/id_w2 
        #Transit frequency of first stage driver
        ft1 = (gmid1 * id_w1)/(2*np.pi*cgg_w1)
        
        #Second current source load stage
        gm5 = 4*np.pi*Cl*fu #Force non-dominant pole to be 2x fu for 60-degree phase margin
        id5 = gm5/gmid5
        W5 = id5/id_w5
        W6 = id5/id_w6
        # Transit frequency of second stage driver
        ft5 = (gmid5 * id_w5) / (2 * np.pi * cgg_w5)

        #4 Calculating Gain and Power
        gds_id1 = gds_w1 / id_w1
        gds_id2 = gds_w2 / id_w2
        gds_id5 = gds_w5 / id_w5
        gds_id6 = gds_w6 / id_w6
        gain_stage1 = gmid1 / np.maximum((gds_id1 + gds_id2), 1e-12)
        gain_stage2 = gmid5 / np.maximum((gds_id5 + gds_id6), 1e-12)
        gain_total = gain_stage1 * gain_stage2

        total_Id = (2*id1) + id5
        
        # 5. Assign Objectives and Constraints
        # Objectives (Minimize)
        f1 = -gain_total  # We want to maximize gain, so we minimize -gain
        f2 = total_Id     # Minimize Power consumption

        # Constraints (Must be <= 0)
        # Constraint 1: ft >= 10GHz -> 10GHz - ft <= 0
        g1 = self.specs['ft_target'] - ft1
        g2 = self.specs['ft_target'] - ft5
        
        max_width = 500e-6 
        g3 = np.maximum(W1 - max_width, W2 - max_width)
        g4 = np.maximum(W5 - max_width, W6 - max_width)
        
        # Constraint 4: Total Current Id must be less than a strict power budget (e.g., 3 mA)
        max_current = 3e-3
        g5 = total_Id - max_current
        g6 = 100 - gain_total

        Vs1 = self.specs['ICM'] - vgs1
        # We demand at least 150mV (0.15V) of headroom for the tail current source.
        # Required math: Vs1 >= 0.15  -->  0.15 - Vs1 <= 0
        min_tail_headroom = 0.15
        g7 = min_tail_headroom - Vs1

        # Update the output dictionary
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6, g7])


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
        'Cc': 1e-12,        # 1 pF
        'VDD': 1.0,         # 1 V
        'gmid2': 10.0,       # PMOS gm/Id
        'gmid5': 10.0,       # PMOS gm/Id
        'ICM': 0.7,
        'VOUT': 0.5         #0.5V
    }

    # 4. Initialize the Problem and Optimizer
    problem = CommonSourceOptimizer(nmos_model, pmos_model, scalers, specs)
    
    algorithm = NSGA2(pop_size=200)
    
    print("Running NSGA-II Optimization...")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 1000),
                   seed=1,
                   verbose=True)

    # 5. Visualize the Pareto Front
    print("\nOptimization Complete. Generating Pareto Front.")
    plot = Scatter(title="Gain vs Current Trade-off (Pareto Front)")
    
    # Invert the objective 1 back to positive Gain for plotting
    f1_gain = -res.F[:, 0]
    f2_id = res.F[:, 1] * 1e6  # Convert to uA
    


    # --- Post-Processing: Extracting Geometries ---
    print("\n--- Extracting Physical Dimensions for First Differential Stage---")
    
    # 1. Choose a design point. 
    # Let's find a balanced design (e.g., a point with reasonable gain and current)
    # Sorting by gain to pick something in the middle of the Pareto front
    sorted_indices = np.argsort(f1_gain)
    chosen_index = sorted_indices[len(sorted_indices) // 2] # Pick the median design
    
    best_X = res.X[chosen_index]
    best_F = res.F[chosen_index]
    
    gmid1_opt = best_X[0]
    gmid6_opt = best_X[1]
    L1_opt = best_X[2]
    L2_opt = best_X[3]
    L6_opt = best_X[4]
    
    print(f"Selected Design - Predicted Gain: {-best_F[0]:.2f} V/V, Total Current: {best_F[1]*1e6:.2f} uA")
    print(f"Optimal L1/L2 (NMOS): {L1_opt * 1e9:.2f} nm")
    print(f"Optimal L3/L4 (PMOS): {L2_opt * 1e9:.2f} nm")
    print(f"Optimal gm/Id M1/M2:  {gmid1_opt:.2f} S/A")
    print(f"Optimal gm/Id M3/M4:  {specs['gmid2']:.2f} S/A")

    # 2. Re-run the ANN for this specific point to get the Widths
    n_params, p_params = resolve_operating_point(gmid1_opt, L1_opt, L2_opt, specs)
    id_wn_opt, gds_wn_opt, vgsn_opt, cgg_wn_opt, cdd_wn_opt, vdsn_opt = n_params
    id_wp_opt, gds_wp_opt, vgsp_opt, cgg_wp_opt, cdd_wp_opt, vdsp_opt = p_params
    
    # 3. Recalculate W1 and W2 
    fu = specs['fu']
    Cl = specs['Cl']
    Cc = specs['Cc']
    gm1 = 2*np.pi*Cc*fu
    id1 = gm1/gmid1_opt
    W1_opt = id1/id_wn_opt
    W2_opt = id1 / id_wp_opt
        
    print(f"Optimal W1/W2 (NMOS): {W1_opt * 1e6:.2f} um")
    print(f"Optimal W2/W3 (PMOS): {W2_opt * 1e6:.2f} um")
    print(f"Optimal VGS1 (NMOS): {vgsn_opt:.2f} V")
    print(f"Optimal VGS2 (PMOS): {abs(vgsp_opt):.2f} V")
    print(f"First stage current: {id1:.2f} V")

    # 4. Sizing the Tail Current Source
    Vs1 = specs['ICM'] - vgsn_opt
    gmid_tail = 15             
        # Now extract the final parameters using the safe gm/Id
    id_wtail, gds_wtail, vgstail, vdsattail, cgg_wtail, cdd_wtail =get_transistor_params(gmid_tail, 400e-9, Vs1, is_nmos=True)
            
    W_tail = (2 * id1) / id_wtail 
    print("\n--- Tail Current Source ---")
    print(f"Available Headroom (Vs1): {Vs1 * 1000:.2f} mV")
    print(f"Tail VDSAT: {vdsattail * 1000:.2f} mV")
    print(f"Optimal gm/Id (Tail): {15:.2f} S/A")
    print(f"Optimal L_tail (NMOS): 400.00 nm")
    print(f"Optimal W_tail (NMOS): {W_tail * 1e6:.2f} um")
    print(f"Optimal VGS_tail (NMOS): {vgstail:.3f} V")

    print("-" * 50)
    #5. Print the paramters for second current source load stage
    print("\n--- Second Current Source Load Stage ---")

    id5_fromtotal = best_F[1] - (2*id1)
    gm5 = 4*np.pi*Cc*fu
    id5_calculated = gm5/specs['gmid5']
    vds5 = specs['VDD'] - specs['VOUT']
    # id_w5, gds_w5, vgs5, vdsat5, cgg_w5, cdd_w5 =get_transistor_params(specs['gmid5'], L2_opt, vds5, is_nmos=False)
    id_w5 = id_wp_opt
    W5_opt = id5_calculated/id_w5
 
    id_w6, gds_w6, vgs6, vdsat6, cgg_w6, cdd_w6 =get_transistor_params(gmid6_opt, L6_opt, specs['VOUT'], is_nmos=True)
    W6_opt = id5_calculated/id_w6

    print(f"From Total Current: {id5_fromtotal*1e6:.2f} uA")
    print(f"Calculated Current: {id5_calculated*1e6:.2f} uA")
    print(f"Optimal gm/id5: {specs['gmid5']:.2f} S/A")
    print(f"Optimal gm/id6: {gmid6_opt:.2f} S/A")
    print(f"Optimal L5 (PMOS): {L2_opt * 1e9:.2f} nm")
    print(f"Optimal W5 (PMOS): {W5_opt * 1e6:.2f} um")
    print(f"Optimal L6 (NMOS): {L6_opt * 1e9:.2f} nm")
    print(f"Optimal W6 (NMOS): {W6_opt * 1e6:.2f} um")
    print(f"Optimal VGS3 (NMOS): {abs(vgs6):.2f} V") 
    print(f"Optimal Vout: {specs['VOUT']:.2f} V")

    plot.add(np.column_stack([f1_gain, f2_id]))
    plot.show()

