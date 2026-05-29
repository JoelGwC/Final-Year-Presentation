import numpy as np
import math
import torch
import joblib
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from model import SurrogateModel
from getparams import get_transistor_params

class RAFFCOptimizer(Problem):
    def __init__(self, nmos_model, pmos_model, scalers, specs):
        # 10 Variables: [gmid1, gmid6, gmid8, gmid12, gmid3, L1, L6, L8, L12, L3]
        # These 10 variables strictly control the 5 physical "layers" of the op-amp
        super().__init__(n_var=12, n_obj=2, n_ieq_constr=6, 
                         xl=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 45e-9, 45e-9, 45e-9, 45e-9, 45e-9, 5.0, 45e-9]), 
                         # Hard-cap L to 440nm to prevent ANN extrapolation hallucinations
                         xu=np.array([15.0, 15.0, 15.0, 15.0, 15.0, 400e-9, 400e-9, 400e-9, 400e-9, 400e-9, 15.0, 400e-9]))
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.scalers = scalers
        self.specs = specs

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        
        # 1. Extract Optimizer Guesses by Layer
        gmid1, gmid6, gmid8, gmid12, gmid3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        L1, L6, L8, L12, L3 = X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        gmid10, L10 = X[:, 10], X[:, 11]

        # 2. Strict Voltage Budget
        vds_fold = np.full(pop_size, specs['vds_fold'])
        vds_out = np.full(pop_size, specs['vds_out'])
        vds_M6 = np.full(pop_size, specs['vds_M6'])
        vds_guess = np.full(pop_size, 0.5)

        # 3. Predict Physical Parameters based on exact VDS for each layer
        # Layer: Top PMOS (M8, M9, M11, M13). VDS = VDD - VOUT = 1.0 - 0.5 = 0.5V
        id_w8, gds_w8, vgs8, vdsat8, cgg_w8, cdd_w8 = get_transistor_params(gmid8, L8, vds_out, is_nmos=False)
        
        # Layer: Middle NMOS Sinks (M12, M14). VDS = VOUT - GND = 0.5V
        id_w12, gds_w12, vgs12, vdsat12, cgg_w12, cdd_w12 = get_transistor_params(gmid12, L12, vds_out, is_nmos=True)

        # Layer: Cascode NMOS (M5, M6). VDS = VOUT1 - Vfold = 0.5 - 0.2 = 0.3V
        id_w6, gds_w6, vgs6, vdsat6, cgg_w6, cdd_w6 = get_transistor_params(gmid6, L6, vds_M6, is_nmos=True)

        # Layer: Bottom NMOS Sinks (M3, M4). VDS = Vfold - GND = 0.2V
        id_w3, gds_w3, vgs3, vdsat3, cgg_w3, cdd_w3 = get_transistor_params(gmid3, L3, vds_fold, is_nmos=True)

        # M10 (Stage 2 Sink) at 0.5V VDS
        id_w10, gds_w10, _, _, _, _ = get_transistor_params(gmid10, L10, vds_out, is_nmos=True)

        # Layer: PMOS Input Pair (M1, M2). 
        # Calculate dynamic VDS based on ICM
        _, _, vgs1_guess, _, _, _ = get_transistor_params(gmid1, L1, vds_guess, is_nmos=False)
        Vs = self.specs['ICM'] + vgs1_guess # vgs1_guess is positive
        vds1_mag = Vs - vds_fold
        id_w1, gds_w1, vgs1, vdsat1, cgg_w1, cdd_w1 = get_transistor_params(gmid1, L1, vds1_mag, is_nmos=False)

        # Force conductances strictly positive to prevent adversarial math glitches
        gds_w1 = np.maximum(gds_w1, id_w1/100)
        gds_w8 = np.maximum(gds_w8, id_w8/100)
        gds_w10 = np.maximum(gds_w10, id_w10/100)
        gds_w12 = np.maximum(gds_w12, id_w12/100)
        
        # 4. Target Specifications
        fu = self.specs['fu']
        Cl = self.specs['Cl']
        Cc1 = self.specs['Cc1']
        PM = self.specs['PM']

        # 5. Stability & Phase Margin Equations (Grasso et al.)
        gm1 = 2 * np.pi * fu * Cc1
        
        # Dynamically calculate exact gmb (M6) required for target Phase Margin
        tan_phi = math.tan(math.radians(PM))
        gm6 = gm1 * (tan_phi + 0.7) 
        
        # Force outer poles to 2*fu for stability
        gm11 = 4 * np.pi * fu * Cl  # gm3 in paper (M11 in schematic)
        gm9 = gm11                  # gm2 in paper (M9 in schematic)
        gm13 = gm11                 # gmf in paper (M13 in schematic)

        # 6. Calculate Currents
        id1 = gm1 / gmid1
        id6 = gm6 / gmid6
        id9 = gm9 / gmid8
        id11 = gm11 / gmid8
        id13 = gm13 / gmid8

        # Branch Currents (For Power Calculation)
        id4 = id1 + id6         # Fold sink M4 carries input branch + cascode
        total_Id = (2 * id4) + id9 + id11 + id13 # Tail + Cascodes + Stage2 + Stage3 + FF

        # 7. Calculate Widths
        W1 = id1 / id_w1
        W6 = id6 / id_w6
        W8 = id6 / id_w8        # M8 supplies M6
        W9 = id9 / id_w8
        W11 = id11 / id_w8
        W13 = id13 / id_w8
        W10 = id9 / id_w10      # M10 sinks Stage 2
        W12 = id11 / id_w12      # M10 sinks Stage 2
        W14 = id13 / id_w12     # M14 sinks Stage 3 and Feedforward

        # 8. Gain Calculation 
        # Using output resistance approximations for sizing limits
        Rout1 = 1.0 / (gds_w8 * W8) # Cascode heavily boosts NMOS, PMOS load dominates
        Rout2 = 1.0 / ((gds_w8 * W9) + (gds_w10 * W10))
        Rout3 = 1.0 / ((gds_w8 * W13) + (gds_w12 * W14))

        gain_total = (gm1 * Rout1) * (gm9 * Rout2) * (gm11 * Rout3)
        
        # 9. Assign Objectives and Constraints
        f1 = -gain_total  
        f2 = total_Id     

        # Constraints (<= 0)
        # fT Constraints to ensure transistors are fast enough
        ft1 = (gmid1 * id_w1) / (2 * np.pi * cgg_w1)
        ft6 = (gmid6 * id_w6) / (2 * np.pi * cgg_w6)
        
        g1 = self.specs['ft_target'] - ft1 
        g2 = self.specs['ft_target'] - ft6 

        # Width constraints to prevent massive layout sizes (> 500um)
        max_width = 500e-6 
        g3 = np.maximum.reduce([W1, W6, W8, W9, W11, W13, W10, W14]) - max_width
        
        # Power budget (Max 3mA)
        g4 = total_Id - 5e-3
        
        # Tail current headroom constraint
        min_tail_headroom = 0.15
        available_headroom = self.specs['VDD'] - Vs
        g5 = min_tail_headroom - available_headroom
        
        # Minimum Gain Constraint (Force at least 1000 V/V for 3 stages)
        g6 = 1000 - gain_total

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6])


if __name__ == "__main__":
    print("Loading pre-trained models and scalers...")
    
    scalers = {
        'X_nmos': joblib.load('scaler_X_nmos.pkl'),
        'y_nmos': joblib.load('scaler_y_nmos.pkl'),
        'X_pmos': joblib.load('scaler_X_pmos.pkl'),
        'y_pmos': joblib.load('scaler_y_pmos.pkl')
    }
    
    nmos_model = SurrogateModel()
    nmos_model.load_state_dict(torch.load('nmos_surrogate_model.pth', weights_only=True))
    nmos_model.eval()
    
    pmos_model = SurrogateModel()
    pmos_model.load_state_dict(torch.load('pmos_surrogate_model.pth', weights_only=True))
    pmos_model.eval()

    # System Specs
    specs = {
        'fu': 10e6,          # 10 MHz
        'ft_target': 10e9,   # Relaxed to 4 GHz
        'VDD': 1.0,          # 1 V
        'PM': 60,           # Target Phase Margin in Degrees
        'Cl': 100e-12,      # 100 pF Load
        'Cc1': 1e-12,       # 1 pF Compensation Cap 1
        'ICM': 0.3,          # 0.3V Input Common Mode
        'gmid0': 10.0,
        'L0': 400e-9,
        'vds_fold': 0.2,
        "vds_out": 0.5,
        "vds_M6": 0.3
    }

    problem = RAFFCOptimizer(nmos_model, pmos_model, scalers, specs)
    algorithm = NSGA2(pop_size=200)
    
    print("Running NSGA-II Optimization...")
    res = minimize(problem, algorithm, ('n_gen', 1000), seed=1, verbose=True)

    # --- Post-Processing: Extracting Geometries ---
    f1_gain = -res.F[:, 0]
    sorted_indices = np.argsort(f1_gain)
    chosen_index = sorted_indices[len(sorted_indices) // 2] 
    
    best_X = res.X[chosen_index]
    best_F = res.F[chosen_index]
    
    gmid1_opt = best_X[0]
    gmid6_opt = best_X[1]
    gmid8_opt = best_X[2]
    gmid12_opt = best_X[3]
    gmid3_opt = best_X[4]
    L1_opt = best_X[5]
    L6_opt = best_X[6]
    L8_opt = best_X[7]
    L12_opt = best_X[8]
    L3_opt = best_X[9]
    gmid10_opt = best_X[10]
    L10_opt = best_X[11]
    
    print(f"\n--- RAFFC Sizing Output ---")
    print(f"Predicted Total DC Gain: {-best_F[0]:.2f} V/V")
    print(f"Predicted Total Current: {best_F[1]*1e6:.2f} uA")
    print("-" * 30)
    
    # Recalculate deterministic Gm values
    fu = specs['fu']
    Cl = specs['Cl']
    Cc1 = specs['Cc1']
    Cl = specs['Cl']
    
    gm1 = 2 * np.pi * fu * Cc1
    tan_phi = math.tan(math.radians(specs['PM']))
    gm6 = gm1 * (tan_phi + 0.7) 
    gm11 = 4 * np.pi * fu * Cl 
    gm9 = gm11
    gm13 = gm11
    
    # Calculate dynamically required Cc2
    Cc2_opt = (2 * gm11 * (Cc1**2)) / (gm6 * Cl)

    # Re-evaluate exactly to extract widths
    _, _, vgs1, _, _, _ = get_transistor_params(gmid1_opt, L1_opt, 0.5, is_nmos=False)
    vs1 = specs['ICM'] + vgs1
    vds1_mag = vs1 - 0.2
    
    id_w1, _, _, _, _, _ = get_transistor_params(gmid1_opt, L1_opt, vds1_mag, is_nmos=False)
    id_w6, _, vgs6, _, _, _ = get_transistor_params(gmid6_opt, L6_opt, 0.3, is_nmos=True)
    id_w8, _, _, _, _, _ = get_transistor_params(gmid8_opt, L8_opt, 0.5, is_nmos=False)
    id_w12, _, _, _, _, _ = get_transistor_params(gmid12_opt, L12_opt, 0.5, is_nmos=True)
    id_w3, _, vgs3, _, _, _ = get_transistor_params(gmid3_opt, L3_opt, 0.2, is_nmos=True)
    id_w10, _, _, _, _, _ = get_transistor_params(gmid10_opt, L10_opt, 0.5, is_nmos=True) # M10 density
    id_w0, _, _, _, _, _ = get_transistor_params(specs['gmid0'], specs['L0'], specs['VDD'] - vs1, is_nmos=False)
    id0 = 2*(gm1/gmid1_opt)


    # Calculate True Branch Currents
    i_stage1 = gm1/gmid1_opt
    i_cascode = gm6/gmid6_opt
    i_stage2 = gm9/gmid8_opt
    i_stage3 = gm11/gmid8_opt
    i_ff = gm13/gmid8_opt

    print(f"Input Current Source (M0) -> gm/Id: {specs['gmid0']:.2f} S/A | L: {specs['L0']*1e9:.2f}nm | W: {(2*i_stage1)/id_w0 * 1e6:.2f}um | Current: {2*i_stage1*1e6:.2f}uA")
    print(f"Source Voltage: {vs1 * 1e3:.2f} mV")
    print(f"Input Pair (M1, M2)  -> gm/Id: {gmid1_opt:.2f} S/A | L: {L1_opt*1e9:.2f}nm | W: {i_stage1/id_w1 * 1e6:.2f}um | Current: {i_stage1*1e6:.2f}uA")
    print(f"Cascodes (M5, M6)    -> gm/Id: {gmid6_opt:.2f} S/A | L: {L6_opt*1e9:.2f}nm | W: {i_cascode/id_w6 * 1e6:.2f}um | Current: {i_cascode*1e6:.2f}uA")
    print(f"Vgs6 (VB3) -> {vgs6*1e3:.2f} mV")
    print(f"Top PMOS (M7, M8)    -> gm/Id: {gmid8_opt:.2f} S/A | L: {L8_opt*1e9:.2f}nm | W: {i_cascode/id_w8 * 1e6:.2f}um | Current: {i_cascode*1e6:.2f}uA")
    print(f"Fold Sinks (M3, M4)  -> gm/Id: {gmid3_opt:.2f} S/A | L: {L3_opt*1e9:.2f}nm | W: {(i_stage1+i_cascode)/id_w3 * 1e6:.2f}um | Current: {(i_stage1+i_cascode)*1e6:.2f}uA")
    print(f"Vgs3 (VB2) -> {vgs3*1e3:.2f} mV")
    
    # BUG FIX: Accurate printing for M10, M12, M14
    print(f"Stage 2 (M9 PMOS)    -> gm/Id: {gmid8_opt:.2f} S/A | L: {L8_opt*1e9:.2f}nm | W: {i_stage2/id_w8 * 1e6:.2f}um | Current: {i_stage2*1e6:.2f}uA")
    print(f"Stage 2 (M10 NMOS)   -> gm/Id: {gmid10_opt:.2f} S/A | L: {L10_opt*1e9:.2f}nm | W: {i_stage2/id_w10 * 1e6:.2f}um | Current: {i_stage2*1e6:.2f}uA")
    
    print(f"Stage 3 (M11 PMOS)   -> gm/Id: {gmid8_opt:.2f} S/A | L: {L8_opt*1e9:.2f}nm | W: {i_stage3/id_w8 * 1e6:.2f}um | Current: {i_stage3*1e6:.2f}uA")
    print(f"Stage 3 (M12 NMOS)   -> gm/Id: {gmid12_opt:.2f} S/A | L: {L12_opt*1e9:.2f}nm | W: {i_stage3/id_w12 * 1e6:.2f}um | Current: {i_stage3*1e6:.2f}uA")
    
    print(f"Stage 3 (M13 FF PMOS)-> gm/Id: {gmid8_opt:.2f} S/A | L: {L8_opt*1e9:.2f}nm | W: {i_ff/id_w8 * 1e6:.2f}um | Current: {i_ff*1e6:.2f}uA")
    print(f"Stage 3 (M14 NMOS)   -> gm/Id: {gmid12_opt:.2f} S/A | L: {L12_opt*1e9:.2f}nm | W: {i_ff/id_w12 * 1e6:.2f}um | Current: {i_ff*1e6:.2f}uA")
    print("-" * 30)
    print(f"Required Cc1: {Cc1 * 1e12:.2f} pF")
    print(f"Calculated Cc2: {Cc2_opt * 1e12:.5f} pF")
    print(f"Calculated CL: {Cl * 1e12:.5f} pF")