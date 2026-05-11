import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def parse_cadence_vcsv(filepath, y_col_name):
    """Parses a Cadence waveVsWave CSV with an ultra-robust multi-line header scanner."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing extracted dataset: {filepath}")
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    if len(lines) < 7:
        raise ValueError(f"File {filepath} appears to be empty or corrupted.")
        
    # Ultra-robust scan across the first 5 lines to find the parameter array
    params = []
    header_line_found = ""
    
    for line in lines[:5]:
        # Matches "L", optional quotes/delimiters, a number, "VDS"/"vds", optional quotes/delimiters, a number
        p = re.findall(r'"?L"?[\s,]+"?([0-9\.eE+-]+)"?[\s,]+"?vds!?"?[\s,]+"?([0-9\.eE+-]+)"?', line, re.IGNORECASE)
        if p:
            params = p
            header_line_found = line
            break
            
    if not params:
        print(f"\n[DEBUG ERROR] Regex failed to match parameters in: {filepath}")
        print("Top 5 lines of the file:")
        for idx, l in enumerate(lines[:5]):
            print(f"Line {idx}: {l.strip()}")
        raise ValueError("Regex extracted 0 parameter pairs. Check your Cadence CSV export delimiters.")
    
    # Load the numerical data, skipping the standard 6 Cadence header rows
    df_raw = pd.read_csv(filepath, skiprows=6, header=None)
    
    frames = []
    for i, (l_val, vds_val) in enumerate(params):
        x_col = i * 2
        y_col = i * 2 + 1
        
        temp_df = df_raw.iloc[:, [x_col, y_col]].copy()
        temp_df.columns = ['gm_Id', y_col_name]
        
        temp_df['L'] = float(l_val)
        temp_df['VDS'] = float(vds_val)
        temp_df['sweep_index'] = temp_df.index 
        
        frames.append(temp_df)
        
    return pd.concat(frames, ignore_index=True).dropna()

def generate_dataset(transistor):
    # Enforce exact directory mapping (NMOS or PMOS in capital letters)
    transistor_dir = transistor.upper()
    print(f"Loading and processing {transistor_dir} characterization data...")

    # Exact paths with no file suffixes
    paths = {
        'Id_W':  f"{transistor_dir}/idW_vs_gmid_vdssweep.vcsv",
        'gds_W': f"{transistor_dir}/gdsW_vs_gmid_vdssweep.vcsv",
        'VGS':   f"{transistor_dir}/vgs_vs_gmid_vdssweep.vcsv",
        'VDSAT': f"{transistor_dir}/vdsat_vs_gmid_vdssweep.vcsv",
        'Cgg_W': f"{transistor_dir}/cggW_vs_gmid_vdssweep.vcsv",
        'Cdd_W': f"{transistor_dir}/cddW_vs_gmid_vdssweep.vcsv"
    }

    # 1. Parse the base file to initialize the master DataFrame
    df_master = parse_cadence_vcsv(paths['Id_W'], 'Id_W')
    
    # 2. Iteratively merge all remaining physical targets
    for col_name in ['gds_W', 'VGS', 'VDSAT', 'Cgg_W', 'Cdd_W']:
        df_temp = parse_cadence_vcsv(paths[col_name], col_name)
        df_master = pd.merge(df_master, df_temp, on=['L', 'VDS', 'sweep_index'], suffixes=('', '_drop'))
        
        if 'gm_Id_drop' in df_master.columns:
            df_master = df_master.drop(columns=['gm_Id_drop'])

    df_master = df_master.drop(columns=['sweep_index'])

    # 3. Filter Analog Operating Bounds (2 S/A to 25 S/A)
    df_master = df_master[(df_master['gm_Id'] >= 2.0) & (df_master['gm_Id'] <= 25.0)]

    # 4. Strictly Handle PMOS Absolute Values
    if transistor.lower() == "pmos":
        print("Applying absolute values to PMOS VGS and VDSAT targets...")
        df_master['VGS'] = df_master['VGS'].abs()
        df_master['VDSAT'] = df_master['VDSAT'].abs()
        
    clean_csv_path = f'{transistor.lower()}_cleaned_complete.csv'
    df_master.to_csv(clean_csv_path, index=False)
    print(f"Clean master dataset compiled: {len(df_master)} valid points. Saved to {clean_csv_path}")



    # 5. Extract ML Features (X) and 6 Output Targets (y)
    X = df_master[['gm_Id', 'L', 'VDS']].values
    y = df_master[['Id_W', 'gds_W', 'VGS', 'VDSAT', 'Cgg_W', 'Cdd_W']].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, f'scaler_X_{transistor.lower()}.pkl')
    joblib.dump(scaler_y, f'scaler_y_{transistor.lower()}.pkl')
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, df_master

def plot_extended_metrics(df, transistor):
    """Generates verification plots for the 6 physical targets."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    colors = df['L'] * 1e9 

    sc0 = axs[0, 0].scatter(df['gm_Id'], df['VDSAT']*1e3, c=colors, cmap='viridis', alpha=0.6)
    axs[0, 0].set_title('Saturation Voltage Limit vs gm/Id')
    axs[0, 0].set_xlabel('gm/Id (S/A)')
    axs[0, 0].set_ylabel('VDSAT (mV)')
    fig.colorbar(sc0, ax=axs[0, 0]).set_label('Length (nm)')

    sc1 = axs[0, 1].scatter(df['gm_Id'], df['gds_W'], c=colors, cmap='plasma', alpha=0.6)
    axs[0, 1].set_title('Output Conductance Density vs gm/Id')
    axs[0, 1].set_xlabel('gm/Id (S/A)')
    axs[0, 1].set_ylabel('gds/W (S/m)')
    fig.colorbar(sc1, ax=axs[0, 1]).set_label('Length (nm)')

    sc2 = axs[1, 0].scatter(df['gm_Id'], df['Cgg_W']*1e9, c=colors, cmap='inferno', alpha=0.6)
    axs[1, 0].set_title('Normalized Gate Capacitance vs gm/Id')
    axs[1, 0].set_xlabel('gm/Id (S/A)')
    axs[1, 0].set_ylabel('Cgg/W (nF/m)')
    fig.colorbar(sc2, ax=axs[1, 0]).set_label('Length (nm)')

    sc3 = axs[1, 1].scatter(df['gm_Id'], df['Cdd_W']*1e9, c=colors, cmap='magma', alpha=0.6)
    axs[1, 1].set_title('Normalized Drain Capacitance vs gm/Id')
    axs[1, 1].set_xlabel('gm/Id (S/A)')
    axs[1, 1].set_ylabel('Cdd/W (nF/m)')
    fig.colorbar(sc3, ax=axs[1, 1]).set_label('Length (nm)')

    plt.tight_layout()
    plt.savefig(f'{transistor.lower()}_extended_metrics.png')
    print(f"Verification plots saved to {transistor.lower()}_extended_metrics.png")
    plt.show()

if __name__ == "__main__":
    # Pass lowercase arguments so the file saving and scalers remain cleanly formatted
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, df_pmos = generate_dataset("pmos")
    plot_extended_metrics(df_pmos, "pmos")
    
    print("-" * 50)
    
    X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n, df_nmos = generate_dataset("nmos")
    plot_extended_metrics(df_nmos, "nmos")