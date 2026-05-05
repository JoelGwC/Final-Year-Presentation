import matplotlib.pyplot as plt
import numpy as np

# 1. Define the data
z = np.linspace(-7, 7, 200)
phi = 1 / (1 + np.exp(-z))

# 2. Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

# 3. Plot the curve in BLACK (as requested)
ax.plot(z, phi, color='black', linewidth=2.5)

# 4. Remove bounding boxes (Top and Right spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5. Configure the visible axes (Left and Bottom)
ax.spines['left'].set_position(('outward', 0))
ax.spines['bottom'].set_position(('outward', 0))

# Add the vertical line at z=0 (characteristic of the original image)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xticks([])
ax.set_yticks([])





# 7. Add Labels and Equation
ax.set_xlabel('z')
ax.set_ylabel(r'$\phi(z)$')
ax.set_title('Softmax', fontsize=16)

# Render the equation using LaTeX syntax
# Placing it roughly where it was in the original image
ax.text(-5, 0.8, r'$\phi(z) = \frac{1}{1 + e^{-z}}$', fontsize=16)

# Show the clean plot
plt.tight_layout()
plt.savefig("softmax.png", dpi=300, bbox_inches="tight")
plt.show()