from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AOI_Data.csv' 

# Read the CSV file

aoi_df = pd.read_csv(file_path)

# Automatically select numerical columns (excluding ID_PCB and Component_ID)

numerical_features = aoi_df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude ID_PCB and Component_ID as they are identifiers, not features

numerical_features = [col for col in numerical_features if col not in ['ID_PCB', 'Component_ID']]

print("Selected Numerical Features:")
print(numerical_features)

scaler = StandardScaler()
aoi_scaled = scaler.fit_transform(aoi_df[numerical_features])

pca = PCA(n_components=0.95)  # Retain 95% of variance
aoi_pca = pca.fit_transform(aoi_scaled)

# Create a new dataframe for PCA results
pca_columns = [f'PC{i+1}' for i in range(aoi_pca.shape[1])]
aoi_pca_df = pd.DataFrame(data=aoi_pca, columns=pca_columns)

# Add ID_PCB and Component_ID back to the dataframe
aoi_pca_df['ID_PCB'] = aoi_df['ID_PCB']
aoi_pca_df['Component_ID'] = aoi_df['Component_ID']

# Save the PCA-transformed data to a CSV file
aoi_pca_df.to_csv('aoi_pca.csv', index=False)

print(aoi_pca_df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')

plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')

plt.grid()
plt.show()

import rainflow

# Apply Rainflow counting to LiftLeadHeight as an example

cycles = rainflow.count_cycles(aoi_df['LiftLeadHeight_1'].values)
print(cycles)