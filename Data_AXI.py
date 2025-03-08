import pandas as pd
import numpy as np

# Load the dataset

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AXI_Data.csv' 

# Read the CSV file

axi_df = pd.read_csv(file_path)

# Identify the base column names

base_columns = ['VoidAreaOfMaxVoidPercent', 'VoidAreaPercent', 'VoidDiameterofMaxVoid']

# Create summary statistics for each base column

summary_stats = {}
for base in base_columns:
    columns = [col for col in axi_df.columns if col.startswith(base)]
    summary_stats[f'{base}_mean'] = axi_df[columns].mean(axis=1)
    summary_stats[f'{base}_max'] = axi_df[columns].max(axis=1)
    summary_stats[f'{base}_min'] = axi_df[columns].min(axis=1)
    summary_stats[f'{base}_std'] = axi_df[columns].std(axis=1)

# Create a new dataframe with summary statistics

axi_summary_df = pd.DataFrame(summary_stats)

# Add ID columns

axi_summary_df['ID_PCB'] = axi_df['ID_PCB']
axi_summary_df['Component_ID'] = axi_df['Component_ID']

# Reorder columns to have ID columns first

cols = axi_summary_df.columns.tolist()
cols = cols[-2:] + cols[:-2]
axi_summary_df = axi_summary_df[cols]

# Now you can proceed with normalization and standardization

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select numerical columns

numerical_cols = axi_summary_df.select_dtypes(include=[np.number]).columns

# Standardization

scaler = StandardScaler()
axi_standardized = axi_summary_df.copy()
axi_standardized[numerical_cols] = scaler.fit_transform(axi_summary_df[numerical_cols])

# Normalization

scaler = MinMaxScaler()
axi_normalized = axi_summary_df.copy()
axi_normalized[numerical_cols] = scaler.fit_transform(axi_summary_df[numerical_cols])

# Visualize the results

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

plt.subplot(131)
sns.boxplot(data=axi_summary_df[numerical_cols])
plt.title('Original Data')
plt.xticks(rotation=90)

plt.subplot(132)
sns.boxplot(data=axi_standardized[numerical_cols])
plt.title('Standardized Data')
plt.xticks(rotation=90)

plt.subplot(133)
sns.boxplot(data=axi_normalized[numerical_cols])
plt.title('Normalized Data')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()