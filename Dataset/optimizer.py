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
        xl = np.array([2.0, 45e-9] * 5 + [1e-6] * 4)
        xu = np.array([25.0, 445e-9] * 5 + [100e-6] * 4)
        
        # Define the problem: 10 variables, 1 objective (Power), 4 constraints
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=4, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        pymoo passes 'X', a 2D array of the entire population's guesses.
        We must evaluate all of them and return objectives (F) and constraints (G).
        """
        pop_size = X.shape[0]
        
        # Arrays to store results for the whole population
        F = np.zeros((pop_size, 1)) # Objective: Minimize Power
        G = np.zeros((pop_size, 4)) # Constraints
        
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
            
            G[i, 0] = 100.0 - gain
            G[i, 1] = 5e6 - gbw
            G[i, 2] = 60.0 - pm
            G[i, 3] = gm1 - gmb # Ensures gmb is strictly greater than gm1

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
    termination = get_termination("n_gen", 100)
    
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
        print("\n--- OPTIMIZATION SUCCESSFUL ---")
        optimal_guesses = res.X
        optimal_power = res.F[0]
        
        print(f"Minimum Power Achieved: {optimal_power * 1e6:.2f} uW")
        
        # Call Phase 3 to calculate final Cadence dimensions!
        bias_currents = optimal_guesses[10:14]
        evaluator.calculate_physical_dimensions(optimal_guesses[:10], bias_currents)
        
    else:
        print("\nOptimization Failed: No design found that satisfies all constraints.")
        print("Try relaxing your target specs (e.g., lower Gain to 80dB or PM to 45°).")