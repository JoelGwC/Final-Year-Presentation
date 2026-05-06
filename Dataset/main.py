import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

#Let user choose whether to parse NMOS or PMOS


def parse_cadence_vcsv(filepath, y_col_name):
    """Parses a Cadence waveVsWave CSV and converts it to a long-format DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines() # Read the file line by line
        
    # Extract L and VDS values from the second line using Regex
    header_line = lines[1]
    params = re.findall(r'"L"\s+([0-9\.eE+-]+)\s+"vds"\s+([0-9\.eE+-]+)', header_line)
    
    # Load the numerical data, skipping the 6 Cadence header rows as they are not needed
    df_raw = pd.read_csv(filepath, skiprows=6, header=None)
    
    frames = []
    for i, (l_val, vds_val) in enumerate(params):
        x_col = i * 2
        y_col = i * 2 + 1
        
        # Isolate the X (gm/Id) and Y data for this specific L and Vds
        temp_df = df_raw.iloc[:, [x_col, y_col]].copy()
        temp_df.columns = ['gm_Id', y_col_name]
        
        # Append the physical parameters
        temp_df['L'] = float(l_val)
        temp_df['VDS'] = float(vds_val)
        
        # Add an index to ensure perfect row-matching when we merge datasets
        temp_df['sweep_index'] = temp_df.index 
        
        frames.append(temp_df)
        
    return pd.concat(frames, ignore_index=True).dropna()


def generate_dataset(transistor):
    print("Parser function loaded successfully!")

    if transistor == "nmos":
        idW_filepath = 'NMOS/idW_vs_gmid_vdssweep.vcsv'
        gain_filepath = 'NMOS/gmgds_vs_gmid_vdssweep.vcsv'
        vgs_filepath = 'NMOS/vgs_vs_gmid_vdssweep.vcsv'
    elif transistor == "pmos":
        idW_filepath = 'PMOS/idW_vs_gmid_vdssweep_pmos.vcsv'
        gain_filepath = 'PMOS/gmgds_vs_gmid_vdssweep_pmos.vcsv'
        vgs_filepath = 'NMOS/vgs_vs_gmid_vdssweep_pmos.vcsv'


    # 1. Parse the files (ensure the filenames match your local directory)
    df_idw = parse_cadence_vcsv(idW_filepath, 'Id_W')
    df_gain = parse_cadence_vcsv(gain_filepath, 'gm_gds')
    # df_vgs = parse_cadence_vcsv(vgs_filepath, 'VGS') # ONLY activate when vgs vs gmid is available

    # 2. Merge them together on L, VDS, and the sweep index
    df_transistor = pd.merge(df_idw, df_gain, on=['L', 'VDS', 'sweep_index'])

    # 3. Clean up the dataframe (drop the redundant sweep_index and duplicate gm_Id_y)
    df_transistor = df_transistor.rename(columns={'gm_Id_x': 'gm_Id'}).drop(columns=['sweep_index', 'gm_Id_y'])

    # 4. Filter the Subthreshold Spaghetti (Noise)
    # We only want realistic analog operating points (gm/Id between 2 and 25)
    df_transistor = df_transistor[(df_transistor['gm_Id'] >= 2.0) & (df_transistor['gm_Id'] <= 25.0)]
    df_transistor.to_csv("cleaned.csv", index=False)

    print(f"Clean NMOS Dataset ready! Total data points: {len(df_transistor)}")
    # print(df_transistor.head())


    # Separate Inputs (Features) and Outputs (Targets)
    # The ANN receives L, VDS, and gm_Id to predict Id_W and gm_gds
    X = df_transistor[['gm_Id', 'L', 'VDS']].values
    y = df_transistor[['Id_W', 'gm_gds']].values 
    # y = df_transistor[['Id_W', 'gm_gds', 'VGS']].values # Uncomment if you want to include VGS

    # Initialize Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform the data
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, f'scaler_X_{transistor}.pkl')
    joblib.dump(scaler_y, f'scaler_y_{transistor}.pkl')
    
    # Split into Training (80%), Validation (10%), and Test (10%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_transistor

#Plotting the cleaned NMOS dataset

def plotGraph(df_transistor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 3. Plot 1: Current Density (Id/W) vs gm/Id
    # We multiply 'L' by 1e9 to convert the colormap label from meters to nanometers
    scatter1 = ax1.scatter(df_transistor['gm_Id'], df_transistor['Id_W'], c=df_transistor['L']*1e9, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('gm/Id (S/A)')
    ax1.set_ylabel('Id/W (A/m)')
    ax1.set_title('Current Density vs Transconductance Efficiency')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Length (nm)')

    # 4. Plot 2: Intrinsic Gain (gm/gds) vs gm/Id
    scatter2 = ax2.scatter(df_transistor['gm_Id'], df_transistor['gm_gds'], c=df_transistor['L']*1e9, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('gm/Id (S/A)')
    ax2.set_ylabel('gm/gds (V/V)')
    ax2.set_title('Intrinsic Gain vs Transconductance Efficiency')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Length (nm)')

    # 5. Format and display the plots
    plt.tight_layout()
    plt.savefig('cleaned.png')
    plt.show()

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, df_transistor = generate_dataset("pmos")
    plotGraph(df_transistor)
