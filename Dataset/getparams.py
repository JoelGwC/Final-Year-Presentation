import numpy as np
import torch
import joblib

# Load scalers
scaler_X_nmos = joblib.load('scaler_X_nmos.pkl')
scaler_y_nmos = joblib.load('scaler_y_nmos.pkl')
scaler_X_pmos = joblib.load('scaler_X_pmos.pkl')
scaler_y_pmos = joblib.load('scaler_y_pmos.pkl')

# Load models (not as strings!)
from model import SurrogateModel

nmos_model = SurrogateModel()
nmos_model.load_state_dict(torch.load('nmos_surrogate_model.pth', weights_only=True))
nmos_model.eval()

pmos_model = SurrogateModel()
pmos_model.load_state_dict(torch.load('pmos_surrogate_model.pth', weights_only=True))
pmos_model.eval()

def get_transistor_params(gm_id, L, vds, is_nmos=True):
    """
    Query surrogate ANN.
    Returns: (id_w, gds_w, vgs, vdsat, cgg_w, cdd_w)
      id_w  [A/m]  — drain current density
      gds_w [S/m]  — output conductance density
      vgs   [V]    — gate-source voltage (signed: negative for PMOS)
      vdsat [V]    — saturation voltage (signed: negative for PMOS)
      cgg_w [F/m]  — total gate cap density
      cdd_w [F/m]  — drain cap density
    """
    # Handle batched inputs (pop_size, 3)
    if isinstance(gm_id, np.ndarray) and len(gm_id.shape) > 0:
        # Ensure inputs are 1D or 2D as needed
        if len(gm_id.shape) == 1:
            raw_inputs = np.column_stack([gm_id, L, vds])
        else:
            raw_inputs = np.column_stack([gm_id, L, vds])
    else:
        # Single input case
        raw_inputs = np.array([[gm_id, L, vds]])
    
    with torch.no_grad():
        if is_nmos:
            scaled_in = scaler_X_nmos.transform(raw_inputs)
            tensor_in = torch.tensor(scaled_in, dtype=torch.float32)
            scaled_out = nmos_model(tensor_in).numpy()
            real_out = scaler_y_nmos.inverse_transform(scaled_out)
        else:
            scaled_in = scaler_X_pmos.transform(raw_inputs)
            tensor_in = torch.tensor(scaled_in, dtype=torch.float32)
            scaled_out = pmos_model(tensor_in).numpy()
            real_out = scaler_y_pmos.inverse_transform(scaled_out)
    
    # Return results in the same format as input
    if len(real_out.shape) > 1 and real_out.shape[0] > 1:
        # Batched case - return tuple of arrays
        return (real_out[:, 0], real_out[:, 1], real_out[:, 2], 
                real_out[:, 3], real_out[:, 4], real_out[:, 5])
    else:
        # Single case - return tuple of scalars
        id_w, gds_w, vgs, vdsat, cgg_w, cdd_w = real_out[0]
        return id_w, gds_w, vgs, vdsat, cgg_w, cdd_w