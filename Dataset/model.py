import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import main


# 1. Define the Architecture (Kept separate so it can be imported elsewhere)
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),      # Inputs: gm/Id, L, VDS
            nn.ReLU(),
            nn.Linear(64, 64),     # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(64, 64),     # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(64, 6)       # Outputs: 'Id_W', 'gds_W', 'VGS', 'VDSAT', 'Cgg_W', 'Cdd_W'
        )
        
    def forward(self, x):
        return self.network(x)

# 2. The Modular Training Function
def train_surrogate(X_train, y_train, X_val, y_val, epochs=500, batch_size=64, force_cpu=False):
    """
    Trains the MLP surrogate model.
    Set force_cpu=True to run entirely on the processor, even if a GPU is present.
    """
    
    # --- Hardware Configuration ---
    if force_cpu:
        device = torch.device("cpu")
        print("Hardware Override: Forcing execution on CPU.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Training initialized on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model detected: {torch.cuda.get_device_name(0)}")

    # --- Data Preparation ---
    # Convert arrays to tensors and send validation data to the selected device
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Model Setup ---
    model = SurrogateModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Send batch to device (CPU or GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()              
            outputs = model(inputs)            
            loss = criterion(outputs, targets) 
            loss.backward()                    
            optimizer.step()                   
            running_loss += loss.item()
            
        # Validation Check
        if (epoch+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {val_loss.item():.6f}")

    end_time = time.time()
    print(f"\nTraining Complete! Total time: {end_time - start_time:.2f} seconds")
    
    return model

# ---------------------------------------------------------
# Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    # Importing X_train, y_train, X_val, y_val from main.py
    print("Loading and preprocessing data...")
    X_train_nmos, X_val_nmos, X_test_nmos, y_train_nmos, y_val_nmos, y_test_nmos, df_nmos = main.generate_dataset("nmos")
    X_train_pmos, X_val_pmos, X_test_pmos, y_train_pmos, y_val_pmos, y_test_pmos, df_pmos = main.generate_dataset("pmos")


    # Set the device to use (CPU or GPU)
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")

    if torch.accelerator.is_available():
        use_cpu = False
        device = "GPU"
    else:
        use_cpu = True
        device = "CPU"

    print(f"--- RUNNING WITH AUTO-DETECT ({device}) ---")

    #Training the model for NMOS
    trained_model_gpu = train_surrogate(X_train_nmos, y_train_nmos, X_val_nmos, y_val_nmos, force_cpu=use_cpu)
    
    # Save the model
    torch.save(trained_model_gpu.state_dict(), 'nmos_surrogate_model.pth')
    print("\nModel for NMOS saved to 'nmos_surrogate_model.pth'")


    #Training the model for PMOS

    print("-" * 50)

    trained_model_gpu = train_surrogate(X_train_pmos, y_train_pmos, X_val_pmos, y_val_pmos, force_cpu=use_cpu)
    
    # Save the model
    torch.save(trained_model_gpu.state_dict(), 'pmos_surrogate_model.pth')
    print("\nModel for PMOS saved to 'pmos_surrogate_model.pth'")
    