# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                   #static data visualization based on matplotlib

# Optional: Increase the precision of displayed float values

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/SPI_Data.csv' 

# Read the CSV file

spi_df = pd.read_csv(file_path)

# Display the first few rows of the dataset

print(spi_df.head())

# Display basic information about the dataset (No. of rows and columns, column names and datatypes, non null count, and memory usage)

print(spi_df.info())

# Generate summary statistics  #new data frame for each column including: no. of non null values, mean, std. dev., min. value, 

print(spi_df.describe())        #25% ~ the value below which 25% of the observations in a dataset fall.

# Check for missing values

# print(spi_df.isnull().sum())        #No missing values in the SPI data set

# Display unique values in categorical columns (In our case these are Component_ID)

categorical_columns = spi_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
     print(f"\nUnique values in {col}:")
     print(spi_df[col].value_counts())

# Identify the base column names

base_columns = ['Height', 'RealArea', 'RealVol']

# Create summary statistics for each base column

summary_stats = {}
for base in base_columns:
    columns = [col for col in spi_df.columns if col.startswith(base)]
    summary_stats[f'{base}_mean'] = spi_df[columns].mean(axis=1)
    summary_stats[f'{base}_max'] = spi_df[columns].max(axis=1)
    summary_stats[f'{base}_min'] = spi_df[columns].min(axis=1)
    summary_stats[f'{base}_std'] = spi_df[columns].std(axis=1)

# Create a new dataframe with summary statistics

spi_summary_df = pd.DataFrame(summary_stats)

# Add ID columns

spi_summary_df['ID_PCB'] = spi_df['ID_PCB']
spi_summary_df['Component_ID'] = spi_df['Component_ID']

# Reorder columns to have ID columns first

cols = spi_summary_df.columns.tolist()
cols = cols[-2:] + cols[:-2]
spi_summary_df = spi_summary_df[cols]

# Now you can proceed with normalization and standardization

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select numerical columns

numerical_cols = spi_summary_df.select_dtypes(include=[np.number]).columns

# Standardization

scaler = StandardScaler()
spi_standardized = spi_summary_df.copy()
spi_standardized[numerical_cols] = scaler.fit_transform(spi_summary_df[numerical_cols])

# Normalization

scaler = MinMaxScaler()
spi_normalized = spi_summary_df.copy()
spi_normalized[numerical_cols] = scaler.fit_transform(spi_summary_df[numerical_cols])

# Visualize the results

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

plt.subplot(131)
sns.boxplot(data=spi_summary_df[numerical_cols])
plt.title('Original Data')
plt.xticks(rotation=90)

plt.subplot(132)
sns.boxplot(data=spi_standardized[numerical_cols])
plt.title('Standardized Data')
plt.xticks(rotation=90)

plt.subplot(133)
sns.boxplot(data=spi_normalized[numerical_cols])
plt.title('Normalized Data')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()