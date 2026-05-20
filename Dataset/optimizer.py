import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import torch

# Import your Evaluator class from Phase 3
from circuit import RAFFC_OpAmp 

class CircuitOptimizer(Problem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        
        # We have 14 variables: 5 pairs of (gm/Id, L) and 4 bias currents
        # Sequence: [gm_id_1, L_1, gm_id_L, L_L, gm_id_2, L_2, gm_id_3, L_3, gm_id_b, L_b, Id_1, Id_2, Id_3, Id_b]
        n_var = 14
        
        # Lower and Upper Bounds
        # gm/Id ranges from 2 to 25 (S/A)
        # L ranges from 45nm to 445nm
        # Bias currents range from 1uA to 100uA
        xl = np.array([10.0, 45e-9] * 5 + [10e-6, 5e-6, 5e-6, 10e-6])
        xu = np.array([30.0, 445e-9] * 5 + [300e-6, 300e-6, 300e-6, 300e-6])
        
        # 1 Objective (Minimize Power + Penalties), 4 Inequalities (G <= 0)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=5, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        pymoo passes 'X', a 2D array of the entire population's guesses.
        We must evaluate all of them and return objectives (F) and constraints (G).
        """
        pop_size = X.shape[0]
        
        # Arrays to store results for the whole population
        F = np.zeros((pop_size, 1)) # Objective: Minimize Power
        G = np.zeros((pop_size, 5)) # Constraints
        
        for i in range(pop_size):
            guesses = X[i, :]
            
            # Phase 3 Evaluator outputs
            # Assuming your evaluate_performance function returns (Gain, GBW, PM, Power)
            gain, gbw, pm, power, CC2 = self.evaluator.evaluate(guesses)
            
            # --- OBJECTIVE ---
            # We want to minimize Power. pymoo always minimizes F.
            F[i, 0] = power
            
            # --- CONSTRAINTS ---
            # pymoo requires constraints to be structured as: G <= 0
            # Target 1: Gain >= 100 dB   ->  100 - Gain <= 0
            # Target 2: GBW >= 5 MHz     ->  5e6 - GBW <= 0
            # Target 3: PM >= 60 degrees ->  60 - PM <= 0
            # Target 4: Asymptotic Stability (gmb > gm1) -> gm1 - gmb <= 0
            
            gm1 = guesses[0] * guesses[10] # gm_id_1 * Id_1
            gmb = guesses[8] * guesses[13] # gm_id_b * Id_b
            
            G[i, 0] = 40.0 - gain
            G[i, 1] = 1e5 - gbw
            G[i, 2] = 60.0 - pm
            G[i, 3] = gm1 - gmb # Ensures gmb is strictly greater than gm1
            # --- NEW 5TH CONSTRAINT: PDK Minimum Width Verification ---
            # Extract ANN current densities to verify physical width W = Id / id_w
            vds_guess = 0.5
            id_w_2, _, _, _, _, _ = self.evaluator.get_transistor_params(guesses[4], guesses[5], vds_guess, is_nmos=False)
            id_w_3, _, _, _, _, _ = self.evaluator.get_transistor_params(guesses[6], guesses[7], vds_guess, is_nmos=False)
            
            W2 = guesses[11] / max(id_w_2, 1e-6) # Stage 2 Width
            W3 = guesses[12] / max(id_w_3, 1e-6) # Stage 3 Width
            
            # Enforce W >= 120nm (1.2e-7 m). Structured as: 1.2e-7 - W <= 0
            min_width_found = min(W2, W3)
            G[i, 4] = 1.2e-7 - min_width_found
        out["F"] = F
        out["G"] = G

if __name__ == "__main__":
    print("Initializing Phase 3 Evaluator...")
    # NOTE: Ensure you pass the correct paths to your trained .pth models and scalers
    # Assuming you have instantiated your nmos_model and loaded the weights:
    evaluator = RAFFC_OpAmp(
        'nmos_surrogate_model.pth', 
        'pmos_surrogate_model.pth', 
        'scaler_X_nmos.pkl', 
        'scaler_y_nmos.pkl',
        'scaler_X_pmos.pkl', 
        'scaler_y_pmos.pkl'
    )
    
    
    problem = CircuitOptimizer(evaluator)
    
    print("Setting up NSGA-II Optimizer...")
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True
    )
    
    # Run the optimization for 100 generations
    termination = get_termination("n_gen", 100000)
    
    print("Starting optimization loop (This may take a minute depending on CPU/GPU)...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        save_history=True,
        verbose=True # Prints progress to the terminal
    )
    
    if res.X is not None:
            print("\n--- OPTIMIZATION SEARCH COMPLETE ---")
            
            # CRITICAL UPDATE: Robustly unpack the absolute best individual if PyMoo returns a 2D front
            if res.X.ndim > 1:
                best_idx = np.argmin(res.F[:, 0])
                optimal_guesses = res.X[best_idx]
                optimal_power = res.F[best_idx, 0]
            else:
                optimal_guesses = res.X
                optimal_power = res.F[0]
                
            # Firewall Verification Check
            if optimal_power >= 1e5:
                print("\n[WARNING] Optimizer converged, but the best design violates DC Saturation Margins!")
                print("The printed dimensions may push intermediate transistors into the triode region.")
                print("Recommendation: Relax your GBW target to 2MHz or decrease the 100dB gain constraint.")
            else:
                print(f"\n[SUCCESS] Valid DC Operating Point Found!")
                print(f"Total Electrical Power Consumption: {optimal_power * 1e6:.2f} uW")
            
            # Extract and compute final Cadence physical blueprint
            bias_currents = optimal_guesses[10:14]
            
            print("\nOptimized Branch Currents:")
            print(f"  Stage 1 (Tail Current M0): {bias_currents[0] * 2 * 1e6:.2f} uA")
            print(f"  Stage 2 (Branch M9/M10):   {bias_currents[1] * 1e6:.2f} uA")
            print(f"  Stage 3 (Branch M11/M12):  {bias_currents[2] * 1e6:.2f} uA")
            print(f"  RAFFC Cascode (M5/M7):     {bias_currents[3] * 1e6:.2f} uA")
            
            evaluator.calculate_physical_dimensions(optimal_guesses[:10], bias_currents)
            
    else:
            print("\nOptimization Failed: No design found that satisfies all performance constraints.")
            print("Try expanding your population size or relaxing target specifications.")