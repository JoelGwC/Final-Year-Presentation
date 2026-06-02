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
        # 16 Variables: [gmid1, gmid6, gmid8, gmid12, gmid3, L1, L6, L8, L12, L3, gmid10, L10, gmid9, L9, gmid11, L11]
        super().__init__(n_var=16, n_obj=2, n_ieq_constr=10, 
                         xl=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 45e-9, 150e-9, 150e-9, 45e-9, 45e-9, 5.0, 150e-9, 5.0, 150e-9, 5.0, 45e-9]), 
                         xu=np.array([15.0, 15.0, 15.0, 15.0, 15.0, 400e-9, 400e-9, 400e-9, 400e-9, 400e-9, 15.0, 400e-9, 15.0, 400e-9, 15.0, 400e-9]))
        
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
        gmid9, L9 = X[:, 12], X[:, 13]
        gmid11, L11 = X[:, 14], X[:, 15]

        # 2. Strict Voltage Budget
        vds_fold = np.full(pop_size, self.specs['vds_fold'])
        vds_out = np.full(pop_size, self.specs['vds_out'])
        vds_guess = np.full(pop_size, 0.5)

        # --------------------------------------------------------------------------------
        # 3. BACKWARD PROPAGATION OF DC BIAS
        # --------------------------------------------------------------------------------
        
        # A. Feedforward Stage (M13, M14) - Tied to VOUT (0.5V)
        id_w14, gds_w14, vgs14, vdsat14, _, _ = get_transistor_params(gmid12, L12, vds_out, is_nmos=True)
        id_w13, gds_w13, vgs13, vdsat13, _, _ = get_transistor_params(gmid9, L9, vds_out, is_nmos=False)
        vgs14 = np.abs(vgs14) # Protect against negative values

        # B. Stage 3 Diode Load (M12)
        id_w12, gds_w12, vgs12seed, vdsat12, _, _ = get_transistor_params(gmid12, L12, vds_out, is_nmos=True)
        id_w12, gds_w12, vgs12, vdsat12, _, _ = get_transistor_params(gmid12, L12, vgs12seed, is_nmos=True)
        vgs12 = np.abs(vgs12)

        # C. Stage 3 Gain Node (M11)
        vds_M11 = np.clip(self.specs['VDD'] - vgs12, 0.05, 0.95)
        id_w11, gds_w11, vgs11, vdsat11, _, _ = get_transistor_params(gmid11, L11, vds_M11, is_nmos=False)
        vgs11 = np.abs(vgs11)

        # D. Stage 2 Node (M9, M10)
        vds_M9 = vds_out
        id_w9, gds_w9, vgs9, vdsat9, _, _ = get_transistor_params(gmid9, L9, vds_M9, is_nmos=False) 
        vgs9 = np.abs(vgs9) # Crucial for the headroom math below!
        
        vds_M10 = self.specs['VDD'] - vds_M9
        id_w10, gds_w10, _, vdsat10, _, _ = get_transistor_params(gmid10, L10, vds_M10, is_nmos=True)

        # E. Stage 1 (Cascode Output) -> Anchored by VGS9
        vds_M8 = np.clip(vgs9, 0.05, 0.95)
        id_w8, gds_w8, vgs8, vdsat8, cgg_w8, cdd_w8 = get_transistor_params(gmid8, L8, vds_M8, is_nmos=False) 
        
        vds_M6 = np.clip((self.specs['VDD'] - vgs9) - self.specs['vds_fold'], 0.05, 0.95)
        id_w6, gds_w6, _, vdsat6, cgg_w6, _ = get_transistor_params(gmid6, L6, vds_M6, is_nmos=True)

        id_w3, gds_w3, _, vdsat3, cgg_w3, _ = get_transistor_params(gmid3, L3, vds_fold, is_nmos=True)
        # --------------------------------------------------------------------------------

        # Layer: PMOS Input Pair (M1, M2). 
        _, _, vgs1_guess, _, _, _ = get_transistor_params(gmid1, L1, vds_guess, is_nmos=False)
        vgs1_guess = np.abs(vgs1_guess)
        Vs = self.specs['ICM'] + vgs1_guess 
        vds1_mag = np.clip(Vs - vds_fold, 0.05, 0.95)
        id_w1, gds_w1, vgs1, vdsat1, cgg_w1, cdd_w1 = get_transistor_params(gmid1, L1, vds1_mag, is_nmos=False)

        # Force conductances strictly positive to prevent adversarial math glitches
        
        id_w1 = np.maximum(np.abs(id_w1), 1e-9)
        id_w3 = np.maximum(np.abs(id_w3), 1e-9)
        id_w6 = np.maximum(np.abs(id_w6), 1e-9)
        id_w8 = np.maximum(np.abs(id_w8), 1e-9)
        id_w9 = np.maximum(np.abs(id_w9), 1e-9)
        id_w10 = np.maximum(np.abs(id_w10), 1e-9)
        id_w11 = np.maximum(np.abs(id_w11), 1e-9)
        id_w12 = np.maximum(np.abs(id_w12), 1e-9)
        id_w13 = np.maximum(np.abs(id_w13), 1e-9)
        id_w14 = np.maximum(np.abs(id_w14), 1e-9)
        
        gds_w1 = np.maximum(gds_w1, id_w1/100)
        gds_w8 = np.maximum(gds_w8, id_w8/100)
        gds_w9 = np.maximum(gds_w9, id_w9/100)
        gds_w10 = np.maximum(gds_w10, id_w10/100)
        gds_w11 = np.maximum(gds_w11, id_w11/100)
        gds_w12 = np.maximum(gds_w12, id_w12/100)
        gds_w13 = np.maximum(gds_w13, id_w13/100)
        gds_w14 = np.maximum(gds_w14, id_w14/100)

        # Absolute value protections for vdsat parameters
        vdsat3 = np.abs(vdsat3)
        vdsat6 = np.abs(vdsat6)
        vdsat8 = np.abs(vdsat8)

        # 4. Target Specifications
        fu = self.specs['fu']
        Cl = self.specs['Cl']
        Cc1 = self.specs['Cc1']
        PM = self.specs['PM']

        # 5. Stability & Phase Margin Equations
        gm1 = 2 * np.pi * fu * Cc1
        
        tan_phi = math.tan(math.radians(PM))
        gm6 = gm1 * (tan_phi + 0.7) 
        
        gm11 = 4 * np.pi * fu * Cl  
        gm9 = gm11                  
        gm13 = gm11                 

        # 6. Calculate Currents
        id1 = gm1 / gmid1
        id6 = gm6 / gmid6
        id9 = gm9 / gmid9
        id11 = gm11 / gmid11
        id13 = gm13 / gmid9

        id4 = id1 + id6         
        total_Id = (2 * id4) + id9 + id11 + id13 

        # 7. Calculate Widths
        W1 = id1 / id_w1
        W6 = id6 / id_w6
        W8 = id6 / id_w8        
        W4 = id4 / id_w3        
        W9 = id9 / id_w9
        W10 = id9 / id_w10      
        W11 = id11 / id_w11
        W12 = id11 / id_w12      
        W13 = id13 / id_w13
        W14 = id13 / id_w14     

        # 8. Gain Calculation 
        # A. True Stage 1 Output Resistance (Cascode Parallel Combination)
        ro8 = 1.0 / (gds_w8 * W8)
        ro2 = 1.0 / (gds_w1 * W1)
        ro4 = 1.0 / (gds_w3 * W4)
        ro6 = 1.0 / (gds_w6 * W6)
        
        R_down = gm6 * ro6 * ((ro2 * ro4) / (ro2 + ro4))
        Rout1 = (ro8 * R_down) / (ro8 + R_down) 
        Rout2 = 1.0 / ((gds_w9 * W9) + (gds_w10 * W10))
        Rout3 = 1.0 / ((gds_w13 * W13) + (gds_w14 * W14))
        current_mirror_ratio = W14 / W12
        
        gm11_effective = gm11 * current_mirror_ratio
        gain_total = (gm1 * Rout1) * (gm9 * Rout2) * (gm11_effective * Rout3)
        
        # 9. Assign Objectives
        f1 = -gain_total  
        f2 = total_Id     

        # 10. Constraints (<= 0)
        ft1 = (gmid1 * id_w1) / (2 * np.pi * cgg_w1)
        ft6 = (gmid6 * id_w6) / (2 * np.pi * cgg_w6)
        
        g1 = self.specs['ft_target'] - ft1 
        g2 = self.specs['ft_target'] - ft6 

        max_width = 500e-6 
        g3 = np.maximum.reduce([W1, W6, W8, W9, W11, W13, W10, W12, W14]) - max_width
        g4 = total_Id - 5e-3
        
        min_tail_headroom = 0.15
        available_headroom = self.specs['VDD'] - Vs
        g5 = min_tail_headroom - available_headroom
        g6 = 1000 - gain_total
        
        v_margin = 0.1 
        g7 = (vdsat3 + v_margin) - vds_fold
        g8 = (vdsat6 + v_margin) - vds_M6
        g9 = (vdsat8 + v_margin) - vds_M8
        
        # FIXED g10: Ensures available headroom supports the cascode + folding sink
        g10 = (vds_fold + vdsat6 + v_margin) - (self.specs['VDD'] - vgs9)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])


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
        'fu': 2e6,          
        'ft_target': 20e6,   
        'VDD': 1.0,          
        'PM': 60,           
        'Cl': 500e-12,      
        'Cc1': 11e-12,       
        'ICM': 0.3,          
        'gmid0': 15.0,
        'L0': 400e-9,
        'vds_fold': 0.2,
        "vds_out": 0.5
        
    }

    problem = RAFFCOptimizer(nmos_model, pmos_model, scalers, specs)
    algorithm = NSGA2(pop_size=400)
    
    print("Running NSGA-II Optimization...")
    res = minimize(problem, algorithm, ('n_gen', 1000), seed=1, verbose=True)

    # --- Post-Processing: Extracting Geometries ---
    f1_gain = -res.F[:, 0]
    sorted_indices = np.argsort(f1_gain)
    chosen_index = sorted_indices[len(sorted_indices) // 2] 
    
    best_X = res.X[chosen_index]
    best_F = res.F[chosen_index]
    
    gmid1_opt, gmid6_opt, gmid8_opt, gmid12_opt, gmid3_opt = best_X[0:5]
    L1_opt, L6_opt, L8_opt, L12_opt, L3_opt = best_X[5:10]
    gmid10_opt, L10_opt = best_X[10], best_X[11]
    gmid9_opt, L9_opt = best_X[12], best_X[13]
    gmid11_opt, L11_opt = best_X[14], best_X[15]
    
    print(f"\n--- RAFFC Sizing Output ---")
    print(f"Predicted Total DC Gain: {-best_F[0]:.2f} V/V")
    print(f"Predicted Total Current: {best_F[1]*1e6:.2f} uA")
    print("-" * 30)
    
    fu, Cl, Cc1 = specs['fu'], specs['Cl'], specs['Cc1']
    
    gm1 = 2 * np.pi * fu * Cc1
    tan_phi = math.tan(math.radians(specs['PM']))
    gm6 = gm1 * (tan_phi + 0.7) 
    gm11 = 4 * np.pi * fu * Cl 
    gm9, gm13 = gm11, gm11
    
    Cc2_opt = (2 * gm11 * (Cc1**2)) / (gm6 * Cl)

    # Re-evaluate exactly matching the Backward Propagation logic
    # Stage 3 backward
    id_w14, _, vgs14_opt, _, _, _ = get_transistor_params(gmid12_opt, L12_opt, specs['vds_out'], is_nmos=True)
    id_w13, _, _, _, _, _ = get_transistor_params(gmid9_opt, L9_opt, specs['vds_out'], is_nmos=False)
    vgs14_opt = np.abs(vgs14_opt)
    
    vds_M12_opt = np.clip(vgs14_opt, 0.05, 0.95)
    id_w12, _, vgs12_opt, _, _, _ = get_transistor_params(gmid12_opt, L12_opt, vds_M12_opt, is_nmos=True)
    vgs12_opt = np.abs(vgs12_opt)
    
    vds_M11_opt = np.clip(specs['VDD'] - vgs12_opt, 0.05, 0.95)
    id_w11, _, vgs11_opt, _, _, _ = get_transistor_params(gmid11_opt, L11_opt, vds_M11_opt, is_nmos=False)
    vgs11_opt = np.abs(vgs11_opt)
    
    # Stage 2 backward
    vds_M9_opt = np.clip(vgs11_opt, 0.05, 0.95)
    id_w9, _, vgs9_opt, _, _, _ = get_transistor_params(gmid9_opt, L9_opt, vds_M9_opt, is_nmos=False) 
    vgs9_opt = np.abs(vgs9_opt)
    
    vds_M10_opt = np.clip(specs['VDD'] - vgs11_opt, 0.05, 0.95)
    id_w10, _, vgs10_opt, _, _, _ = get_transistor_params(gmid10_opt, L10_opt, vds_M10_opt, is_nmos=True)

    # Stage 1 backward
    vds_M8_opt = np.clip(vgs9_opt, 0.05, 0.95)
    id_w8, _, vgs8_opt, _, _, _ = get_transistor_params(gmid8_opt, L8_opt, vds_M8_opt, is_nmos=False)
    vgs8_opt = np.abs(vgs8_opt)

    vds_M6_opt = np.clip((specs['VDD'] - vgs9_opt) - specs['vds_fold'], 0.05, 0.95)
    id_w6, _, vgs6_opt, _, _, _ = get_transistor_params(gmid6_opt, L6_opt, vds_M6_opt, is_nmos=True)
    
    id_w3, _, vgs3_opt, _, _, _ = get_transistor_params(gmid3_opt, L3_opt, specs['vds_fold'], is_nmos=True)

    # Stage 1 Input
    _, _, vgs1_guess, _, _, _ = get_transistor_params(gmid1_opt, L1_opt, 0.5, is_nmos=False)
    vs1_opt = specs['ICM'] + np.abs(vgs1_guess)
    vds1_mag_opt = np.clip(vs1_opt - specs['vds_fold'], 0.05, 0.95)
    id_w1, _, _, _, _, _ = get_transistor_params(gmid1_opt, L1_opt, vds1_mag_opt, is_nmos=False)
    id_w0, _, _, _, _, _ = get_transistor_params(specs['gmid0'], specs['L0'], specs['VDD'] - vs1_opt, is_nmos=False)

    # Calculate True Branch Currents
    i_stage1 = gm1/gmid1_opt
    i_cascode = gm6/gmid6_opt
    i_stage2 = gm9/gmid9_opt
    i_stage3 = gm11/gmid11_opt
    i_ff = gm13/gmid9_opt

    print(f"Input Current Source (M0) -> gm/Id: {specs['gmid0']:.2f} S/A | L: {specs['L0']*1e9:.2f}nm | W: {(2*i_stage1)/id_w0 * 1e6:.2f}um | Current: {2*i_stage1*1e6:.2f}uA")
    print(f"Source Voltage: {vs1_opt * 1e3:.2f} mV")
    print(f"Input Pair (M1, M2)  -> gm/Id: {gmid1_opt:.2f} S/A | L: {L1_opt*1e9:.2f}nm | W: {i_stage1/id_w1 * 1e6:.2f}um | Current: {i_stage1*1e6:.2f}uA")
    print(f"Cascodes (M5, M6)    -> gm/Id: {gmid6_opt:.2f} S/A | L: {L6_opt*1e9:.2f}nm | W: {i_cascode/id_w6 * 1e6:.2f}um | Current: {i_cascode*1e6:.2f}uA")
    print(f"Vgs6 (VB3) -> {vgs6_opt*1e3:.2f} mV")
    print(f"Top PMOS (M7, M8)    -> gm/Id: {gmid8_opt:.2f} S/A | L: {L8_opt*1e9:.2f}nm | W: {i_cascode/id_w8 * 1e6:.2f}um | Current: {i_cascode*1e6:.2f}uA")
    print(f"VDS8: {vds_M8_opt * 1e3:.2f} mV")
    print(f"VGS8: {vgs8_opt * 1e3:.2f} mV")
    print(f"VGS9: {vgs9_opt * 1e3:.2f} mV")
    print(f"VGS12: {vgs12_opt * 1e3:.2f} mV")
    print(f"VGS14: {vgs14_opt * 1e3:.2f} mV")
    print(f"Fold Sinks (M3, M4)  -> gm/Id: {gmid3_opt:.2f} S/A | L: {L3_opt*1e9:.2f}nm | W: {(i_stage1+i_cascode)/id_w3 * 1e6:.2f}um | Current: {(i_stage1+i_cascode)*1e6:.2f}uA")
    print(f"Vgs3 (VB2) -> {vgs3_opt*1e3:.2f} mV")
    
    print(f"Stage 2 (M9 PMOS)    -> gm/Id: {gmid9_opt:.2f} S/A | L: {L9_opt*1e9:.2f}nm | W: {i_stage2/id_w9 * 1e6:.2f}um | Current: {i_stage2*1e6:.2f}uA")
    print(f"Stage 2 (M10 NMOS)   -> gm/Id: {gmid10_opt:.2f} S/A | L: {L10_opt*1e9:.2f}nm | W: {i_stage2/id_w10 * 1e6:.2f}um | Current: {i_stage2*1e6:.2f}uA")
    print(f"VB4 (M10 NMOS)       -> {vgs10_opt*1e3:.2f} mV")
    
    print(f"Stage 3 (M11 PMOS)   -> gm/Id: {gmid11_opt:.2f} S/A | L: {L11_opt*1e9:.2f}nm | W: {i_stage3/id_w11 * 1e6:.2f}um | Current: {i_stage3*1e6:.2f}uA")
    print(f"Stage 3 (M12 NMOS)   -> gm/Id: {gmid12_opt:.2f} S/A | L: {L12_opt*1e9:.2f}nm | W: {i_stage3/id_w12 * 1e6:.2f}um | Current: {i_stage3*1e6:.2f}uA")
    
    print(f"Stage 3 (M13 FF PMOS)-> gm/Id: {gmid9_opt:.2f} S/A | L: {L9_opt*1e9:.2f}nm | W: {i_ff/id_w13 * 1e6:.2f}um | Current: {i_ff*1e6:.2f}uA")
    print(f"Stage 3 (M14 NMOS)   -> gm/Id: {gmid12_opt:.2f} S/A | L: {L12_opt*1e9:.2f}nm | W: {i_ff/id_w14 * 1e6:.2f}um | Current: {i_ff*1e6:.2f}uA")
    print("-" * 30)
    print(f"Required Cc1: {Cc1 * 1e12:.2f} pF")
    print(f"Calculated Cc2: {Cc2_opt * 1e12:.5f} pF")
    print(f"Calculated CL: {Cl * 1e12:.5f} pF")