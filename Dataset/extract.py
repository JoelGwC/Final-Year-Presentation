import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Cadence output
# sep='\s+' tells pandas to look for any amount of whitespace between columns
# skiprows=1 ignores the messy Cadence equation header
# names=[...] assigns clean, usable column names
df = pd.read_csv('NMOS.txt', sep='\s+', skiprows=1, names=['VGS', 'gm_id'])

# 2. Print the first 5 rows to verify it loaded correctly
print("Data Preview:")
print(df.head())

# 3. Plot the data to confirm it matches your Cadence ViVA graph
plt.figure(figsize=(8, 5))
plt.plot(df['VGS'], df['gm_id'], color='cyan', linewidth=2)
plt.title('Transconductance Efficiency ($g_m/I_D$) vs $V_{GS}$')
plt.xlabel('$V_{GS}$ (V)')
plt.ylabel('$g_m/I_D$ (S/A)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()