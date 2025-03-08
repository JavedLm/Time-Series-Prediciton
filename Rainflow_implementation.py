import pandas as pd
import rainflow
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('C:/Users/Javed Khan/Downloads/Simulink-Makro_neuStepSize/Ergebnisse_Simulink-Makro_0.18-W_Stadt_Berufspendler_Sommer.csv')

# Perform rainflow counting on the 'Tj' column
cycles = rainflow.count_cycles(df['Tj'])

# Extract cycle ranges and counts
ranges, counts = zip(*cycles)

# Convert to numpy arrays for easier manipulation
ranges = np.array(ranges)
counts = np.array(counts)

# Filter out very small cycles (adjust the threshold as needed)
threshold = 0.001  # 0.001°C
mask = ranges > threshold
ranges_filtered = ranges[mask]
counts_filtered = counts[mask]

# Plot the results
plt.figure(figsize=(12, 6))
plt.bar(ranges_filtered, counts_filtered, width=np.min(ranges_filtered)*0.8, edgecolor='black')
plt.xlabel('Temperature Cycle Range (°C)')
plt.ylabel('Cycle Count')
plt.title('Rainflow Counting Results for Junction Temperature (Tj)')
plt.xscale('log')  # Use logarithmic scale for x-axis
plt.grid(True)
plt.show()

# Print summary statistics
print(f"Total cycles: {np.sum(counts_filtered)}")
print(f"Min cycle range: {np.min(ranges_filtered):.6f}°C")
print(f"Max cycle range: {np.max(ranges_filtered):.6f}°C")
print(f"Mean cycle range: {np.mean(ranges_filtered):.6f}°C")

# Print top 10 most frequent cycles
sorted_indices = np.argsort(counts_filtered)[::-1]
print("\nTop 10 most frequent cycles:")
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    print(f"Range: {ranges_filtered[idx]:.6f}°C, Count: {counts_filtered[idx]:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['Tj'])
plt.xlabel('Time (s)')
plt.ylabel('Junction Temperature (°C)')
plt.title('Junction Temperature Over Time')
plt.grid(True)
plt.show()

# Calculate a simplified power factor (normalized power)
df['PowerFactor'] = df['Power'] / df['Power'].max()

# Perform rainflow counting on PowerFactor column
cycles_pf = rainflow.count_cycles(df['PowerFactor'])

# Perform rainflow counting on Tj column
cycles_tj = rainflow.count_cycles(df['Tj'])

# Extract cycle ranges and counts for Tj
ranges_tj, counts_tj = zip(*cycles_tj)
ranges_tj, counts_tj = np.array(ranges_tj), np.array(counts_tj)

# Extract cycle ranges and counts for PowerFactor
ranges_pf, counts_pf = zip(*cycles_pf)
ranges_pf, counts_pf = np.array(ranges_pf), np.array(counts_pf)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Temperature plot
ax1.bar(ranges_tj, counts_tj, width=np.min(ranges_tj)*0.8, edgecolor='black')
ax1.set_xlabel('Temperature Cycle Range (°C)')
ax1.set_ylabel('Cycle Count')
ax1.set_title('Rainflow Counting Results for Junction Temperature (Tj)')
ax1.set_xscale('log')
ax1.grid(True)

# Power Factor plot
ax2.bar(ranges_pf, counts_pf, width=np.min(ranges_pf)*0.8, edgecolor='black')
ax2.set_xlabel('Power Factor Cycle Range')
ax2.set_ylabel('Cycle Count')
ax2.set_title('Rainflow Counting Results for Power Factor')
ax2.set_xscale('log')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print summary statistics for both Tj and Power Factor
print("Temperature (Tj) Statistics:")
print(f"Total cycles: {np.sum(counts_tj)}")
print(f"Min cycle range: {np.min(ranges_tj):.6f}°C")
print(f"Max cycle range: {np.max(ranges_tj):.6f}°C")
print(f"Mean cycle range: {np.mean(ranges_tj):.6f}°C")

print("\nPower Factor Statistics:")
print(f"Total cycles: {np.sum(counts_pf)}")
print(f"Min cycle range: {np.min(ranges_pf):.6f}")
print(f"Max cycle range: {np.max(ranges_pf):.6f}")
print(f"Mean cycle range: {np.mean(ranges_pf):.6f}")