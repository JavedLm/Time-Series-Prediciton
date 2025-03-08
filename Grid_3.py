import pandas as pd
import numpy as np
import rainflow
import matplotlib.pyplot as plt
from collections import defaultdict

# Step 1: Load and prepare the data
df = pd.read_csv('C:/Users/Javed Khan/Downloads/Simulink-Makro_neuStepSize/Ergebnisse_Simulink-Makro_0.5-W_Wochenendenfahrzeug_Winter.csv')  # Replace with your actual CSV file name

print(df['Power'].describe())

plt.figure(figsize=(12, 6))

# Plot the power values
plt.plot(df['Power'].values, color='blue', linewidth=0.5)

# Set labels and title
plt.xlabel('Data Point Index')
plt.ylabel('Power')
plt.title('Variation of Power Values')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust y-axis to focus on the variations
power_mean = df['Power'].mean()
power_std = df['Power'].std()
y_min = max(df['Power'].min(), power_mean - 5*power_std)
y_max = min(df['Power'].max(), power_mean + 5*power_std)
plt.ylim(y_min, y_max)

# Add text box with statistics
stats = f"Mean: {power_mean:.12f}\nStd Dev: {power_std:.12f}\nMin: {df['Power'].min():.12f}\nMax: {df['Power'].max():.12f}"
plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print the unique values in Power to see the level of variation
print("\nUnique Power values:")
print(np.sort(df['Power'].unique()))

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# # Plot 2: Zoomed in view of Power variations
# mean_power = df['Power'].mean()
# std_power = df['Power'].std()
# y_min = mean_power - 5*std_power
# y_max = mean_power + 5*std_power

# ax2.plot(df.index, df['Power'], color='green', linewidth=0.5)
# ax2.set_title('Power Values (Zoomed)')
# ax2.set_xlabel('Data Point Index')
# ax2.set_ylabel('Power')
# ax2.set_ylim(y_min, y_max)
# ax2.grid(True)

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.show()

# # Print detailed statistics
# print(f"Power range: {df['Power'].min():.14f} to {df['Power'].max():.14f}")
# print(f"Power mean: {df['Power'].mean():.14f}")
# print(f"Power standard deviation: {df['Power'].std():.14f}")