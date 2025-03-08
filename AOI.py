# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                   #static data visualization based on matplotlib

# Optional: Increase the precision of displayed float values

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AOI_Data.csv' 

# Read the CSV file

df = pd.read_csv(file_path)

# Display the first few rows of the dataset

# print(df.head())

# Display basic information about the dataset (No. of rows and columns, column names and datatypes, non null count, and memory usage)

print(df.info())

# Generate summary statistics  #new data frame for each column including: no. of non null values, mean, std. dev., min. value, 

print(df.describe())        #25% ~ the value below which 25% of the observations in a dataset fall.

# Check for missing values

# print(df.isnull().sum())        #No missing values in the AOI data set

# Display unique values in categorical columns (In our case these are Component_ID)

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
     print(f"\nUnique values in {col}:")
     print(df[col].value_counts())

# Step 6: Normalization and Standardization

# Import necessary scaling libraries

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # Identify numerical columns

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Method 1: StandardScaler (Z-score normalization)
# Transforms data to have mean=0 and standard deviation=1

scaler_standard = StandardScaler()
df_standardized = df.copy()
df_standardized[numerical_columns] = scaler_standard.fit_transform(df[numerical_columns])

# Method 2: MinMaxScaler (Normalization)
# Scales features to a fixed range, typically between 0 and 1

scaler_minmax = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numerical_columns] = scaler_minmax.fit_transform(df[numerical_columns])

# # Visualize the distribution after standardization

plt.figure(figsize=(15, 5))
plt.subplot(121)
df_standardized[numerical_columns].boxplot()
plt.title('Standardized Features')
plt.xticks(rotation=45)

plt.subplot(122)
df_normalized[numerical_columns].boxplot()
plt.title('Normalized Features')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()