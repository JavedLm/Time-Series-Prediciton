import pandas as pd
import numpy as np
import rainflow
import matplotlib.pyplot as plt
from collections import defaultdict

# Step 1: Load and prepare the data
df = pd.read_csv('C:/Users/Javed Khan/Downloads/Simulink-Makro_neuStepSize/Ergebnisse_Simulink-Makro_0.5-W_Wochenendenfahrzeug_Winter.csv')
print(df.head())
print(df.info())

# Create a figure with three subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15), sharex=True)

# Plot Time vs Junction_Temps (Junction Temperature)
ax1.plot(df['Time'], df['Junction_Temps'], color='red')
ax1.set_ylabel('Junction Temperature (°C)')
ax1.set_title('Junction Temperature vs Time')
ax1.grid(True)

# Plot Time vs Power

power_mean = df['Power'].mean()
power_std = df['Power'].std()
y_min = max(df['Power'].min(), power_mean - 3*power_std)
y_max = min(df['Power'].max(), power_mean + 3*power_std)

ax2.plot(df['Time'], df['Power'], color='green')
ax2.set_ylabel('Power (Zoomed)')
ax2.set_title('Power vs Time (Zoomed)')
ax2.set_ylim(y_min, y_max)
ax2.grid(True)

# Set the x-axis label for the bottom subplot
ax2.set_xlabel('Time (s)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print some statistics
print(f"Time range: {df['Time'].min()} to {df['Time'].max()} seconds")
print(f"Junction_Temps range: {df['Junction_Temps'].min():.2f} to {df['Junction_Temps'].max():.2f} °C")
print(f"Power range: {df['Power'].min():.12f} to {df['Power'].max():.12f}")






# Step 2: Apply Rainflow counting to each column & Extract features and create time stamps

def apply_rainflow(data):
    return list(rainflow.extract_cycles(data))


def apply_rainflow(data):
    return list(rainflow.extract_cycles(data))

def extract_features(cycles, column_name):
    features = []
    for i, cycle in enumerate(cycles):
        feature = {
            'Column': column_name,
            'Cycle_ID': i,
            'Range': cycle[0],
            'Mean': cycle[1],
            'Start': cycle[2],
            'End': cycle[3] if len(cycle) > 3 else None,
            'Count': cycle[4] if len(cycle) > 4 else 1.0
        }
        features.append(feature)
    return pd.DataFrame(features)

# Apply Rainflow counting to Junction_Temps and Power

tj_cycles = apply_rainflow(df['Junction_Temps'].values)
power_cycles = apply_rainflow(df['Power'].values)

tj_features = extract_features(tj_cycles, 'Junction_Temps')
power_features = extract_features(power_cycles, 'Power')

# Combine features
all_features = pd.concat([tj_features, power_features], ignore_index=True)

# Display results
print("Rainflow Counting Results:")
print(all_features)

# Visualize the cycles
plt.figure(figsize=(15, 10))

plt.subplot(1, 1, 1)
plt.scatter(tj_features['Mean'], tj_features['Range'], alpha=0.5)
plt.title('Junction Temperature Cycles')
plt.xlabel('Mean Temperature')
plt.ylabel('Temperature Range')

plt.tight_layout()
plt.show()

# Save results
all_features.to_csv('rainflow_results.csv', index=False)
print("Rainflow results saved to 'rainflow_results.csv'")











time_cycles = apply_rainflow(df['Time'].values)
tj_cycles = apply_rainflow(df['Junction_Temps'].values)
power_cycles = apply_rainflow(df['Power'].values)

def extract_features(cycles, column_name):
    features = []
    for cycle in cycles:
        feature = {
            'Column': column_name,
            'Range': cycle[0],
            'Mean': cycle[1],
            'Count': 1.0,  # Assuming full cycles, adjust if needed
            'Start_Index': cycle[2] if len(cycle) > 2 else None,
            'End_Index': cycle[3] if len(cycle) > 3 else None,
            'Amplitude': cycle[0] / 2,
            'Duration': cycle[3] - cycle[2] if len(cycle) > 3 and cycle[3] > cycle[2] else None
        }
        features.append(feature)
    return pd.DataFrame(features)

# Apply Rainflow counting and extract features
time_cycles = apply_rainflow(df['Time'].values)
tj_cycles = apply_rainflow(df['Junction_Temps'].values)
power_cycles = apply_rainflow(df['Power'].values)

time_features = extract_features(time_cycles, 'Time')
tj_features = extract_features(tj_cycles, 'Junction_Temps')
power_features = extract_features(power_cycles, 'Power')

# Combine all features
all_features = pd.concat([time_features, tj_features, power_features], ignore_index=True)
print(all_features.head(15))

# Save results to CSV
all_features.to_csv('rainflow_features.csv', index=False)
print("Rainflow features saved to 'rainflow_features.csv'")

# # Alternative step: plot the raw data (without the use of rainflow counting algorithm)

# plt.figure(figsize=(15, 10))

# plt.subplot(2, 1, 1)
# plt.plot(df['Time'], df['Junction_Temps'])
# plt.title('Junction Temperature over Time')
# plt.xlabel('Time')
# plt.ylabel('Junction Temperature')

# # Calculate and plot the rate of change
# df['Temp_Rate'] = df['Junction_Temps'].diff() / df['Time'].diff()
# df['Power_Rate'] = df['Power'].diff() / df['Time'].diff()

# plt.subplot(2, 1, 2)
# plt.plot(df['Time'][1:], df['Temp_Rate'][1:], label='Temperature Rate')
# plt.plot(df['Time'][1:], df['Power_Rate'][1:], label='Power Rate')
# plt.title('Rate of Change over Time')
# plt.xlabel('Time')
# plt.ylabel('Rate of Change')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Calculate summary statistics
# summary = df.describe()
# print(summary)

# # Identify significant changes
# temp_threshold = df['Temp_Rate'].std() * 2
# power_threshold = df['Power_Rate'].std() * 2

# significant_changes = df[(abs(df['Temp_Rate']) > temp_threshold) | (abs(df['Power_Rate']) > power_threshold)]
# print("\nSignificant Changes:")
# print(significant_changes)

# # Save processed data
# df.to_csv('processed_data.csv', index=False)
# print("\nProcessed data saved to 'processed_data.csv'")


# # Step 3: Extract features and create time stamps

# def extract_features(cycles, Time):
#     features = []
#     for i, cycle in enumerate(cycles):
#         feature = {
#             'Column': Time,
#             'Cycle_ID': i,
#         }
#         if len(cycle) >= 4:
#             feature.update({
#                 'Range': cycle[0],
#                 'Mean': cycle[1],
#                 'Start_Index': cycle[2],
#                 'End_Index': cycle[3],
#                 'Amplitude': cycle[0] / 2,  # Range is peak-to-peak, so divide by 2 for amplitude
#                 'Duration': cycle[3] - cycle[2] if cycle[3] > cycle[2] else None
#             })
#         else:
#             feature.update({
#                 'Range': cycle[0] if len(cycle) > 0 else None,
#                 'Mean': cycle[1] if len(cycle) > 1 else None,
#                 'Start_Index': None,
#                 'End_Index': None,
#                 'Amplitude': cycle[0] / 2 if len(cycle) > 0 else None,
#                 'Duration': None
#             })
#         features.append(feature)
#     return pd.DataFrame(features)

# time_features = extract_features(time_cycles, 'Time')
# tj_features = extract_features(tj_cycles, 'Junction_Temps')
# power_features = extract_features(power_cycles, 'Power')

# all_features = pd.concat([time_features, tj_features, power_features], ignore_index=True)
# print(all_features.head())






# # Step 4: Calculate duty cycles
# def calculate_duty_cycles(data, column, threshold):
#     total_time = len(data)
#     time_above_threshold = len(data[data[column] > threshold])
#     duty_cycle = (time_above_threshold / total_time) * 100
#     return duty_cycle

# time_duty_cycle = calculate_duty_cycles(df, 'Time', df['Time'].mean())
# tj_duty_cycle = calculate_duty_cycles(df, 'Junction_Temps', df['Junction_Temps'].mean())
# power_duty_cycle = calculate_duty_cycles(df, 'Power', df['Power'].mean())

# print(f"Duty Cycle (Time): {time_duty_cycle:.2f}%")
# print(f"Duty Cycle (Junction_Temps): {tj_duty_cycle:.2f}%")
# print(f"Duty Cycle (Power): {power_duty_cycle:.2f}%")

# # Step 5: Implement grid magic
# def create_grid(features, amplitude_bins, duration_bins):
#     grid = defaultdict(int)
#     for _, cycle in features.iterrows():
#         amp_bin = np.digitize(cycle['Amplitude'], amplitude_bins)
#         dur_bin = np.digitize(cycle['Duration'], duration_bins)
#         grid[(amp_bin, dur_bin)] += 1
#     return grid

# # Define bins for amplitude and duration
# amplitude_bins = np.linspace(0, all_features['Amplitude'].max(), 10)
# duration_bins = np.linspace(0, all_features['Duration'].max(), 10)

# time_grid = create_grid(time_features, amplitude_bins, duration_bins)
# tj_grid = create_grid(tj_features, amplitude_bins, duration_bins)
# power_grid = create_grid(power_features, amplitude_bins, duration_bins)

# # Visualize grid magic
# def plot_grid(grid, title):
#     grid_array = np.zeros((len(amplitude_bins), len(duration_bins)))
#     for (i, j), count in grid.items():
#         grid_array[i, j] = count
    
#     plt.figure(figsize=(10, 8))
#     plt.imshow(grid_array, cmap='viridis', aspect='auto')
#     plt.colorbar(label='Count')
#     plt.title(title)
#     plt.xlabel('Duration Bin')
#     plt.ylabel('Amplitude Bin')
#     plt.show()

# plot_grid(time_grid, 'Time Grid Magic')
# plot_grid(tj_grid, 'Junction_Temps Grid Magic')
# plot_grid(power_grid, 'Power Grid Magic')

# # Step 6: Implement bucket requirement
# def create_buckets(features, column, num_buckets):
#     min_val = features[column].min()
#     max_val = features[column].max()
#     bucket_edges = np.linspace(min_val, max_val, num_buckets + 1)
    
#     buckets = defaultdict(lambda: {'count': 0, 'total_amplitude': 0, 'total_duration': 0})
#     for _, cycle in features.iterrows():
#         bucket = np.digitize(cycle[column], bucket_edges) - 1
#         buckets[bucket]['count'] += 1
#         buckets[bucket]['total_amplitude'] += cycle['Amplitude']
#         buckets[bucket]['total_duration'] += cycle['Duration']
    
#     for bucket in buckets.values():
#         bucket['avg_amplitude'] = bucket['total_amplitude'] / bucket['count'] if bucket['count'] > 0 else 0
#         bucket['avg_duration'] = bucket['total_duration'] / bucket['count'] if bucket['count'] > 0 else 0
    
#     return buckets, bucket_edges

# time_buckets, time_edges = create_buckets(time_features, 'Amplitude', 5)
# tj_buckets, tj_edges = create_buckets(tj_features, 'Amplitude', 5)
# power_buckets, power_edges = create_buckets(power_features, 'Amplitude', 5)

# # Print bucket information
# def print_bucket_info(buckets, edges, title):
#     print(f"\n{title} Buckets:")
#     for i, (edge_low, edge_high) in enumerate(zip(edges[:-1], edges[1:])):
#         bucket = buckets[i]
#         print(f"Bucket [{edge_low:.2f}, {edge_high:.2f}]: "
#               f"Count: {bucket['count']}, "
#               f"Avg Amplitude: {bucket['avg_amplitude']:.2f}, "
#               f"Avg Duration: {bucket['avg_duration']:.2f}")

# print_bucket_info(time_buckets, time_edges, "Time")
# print_bucket_info(tj_buckets, tj_edges, "Junction_Temps")
# print_bucket_info(power_buckets, power_edges, "Power")

# # Save results to CSV
# all_features.to_csv('rainflow_features.csv', index=False)
# print("Rainflow features saved to 'rainflow_features.csv'")
