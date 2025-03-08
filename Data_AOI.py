# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                   #static data visualization based on matplotlib

# # Optional: Set plot style for better visualizations

# plt.style.use('seaborn')

# # Optional: Set pandas to display all columns

# pd.set_option('display.max_columns', None)

# # Optional: Increase the precision of displayed float values

# pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AOI_Data.csv' 

# Read the CSV file

df = pd.read_csv(file_path)

# Display the first few rows of the dataset
# print(df.head())

# Display basic information about the dataset (No. of rows and columns, column names and datatypes, non null count, and memory usage)
# print(df.info())

# Generate summary statistics  #new data frame for each column including: no. of non null values, mean, std. dev., min. value, 
# print(df.describe())        #25% ~ the value below which 25% of the observations in a dataset fall.

# Check for missing values
# print(df.isnull().sum())        #No missing values in the AOI data set


# Display unique values in categorical columns (if any)

# categorical_columns = df.select_dtypes(include=['object']).columns
# for col in categorical_columns:
#      print(f"\nUnique values in {col}:")
#      print(df[col].value_counts())                                #Frequency of a particular value in the dataset (count)

# Visualize the distribution of numerical features

# numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# fig, axes = plt.subplots(nrows=(len(numerical_columns)+1)//2, ncols=2, figsize=(15, 4*((len(numerical_columns)+1)//2)))

# fig.suptitle('Distribution of Numerical Features', fontsize=16)

# for i, col in enumerate(numerical_columns):
#       sns.histplot(df[col], kde=True, ax=axes[i//2, i%2])
#       axes[i//2, i%2].set_title(col)

# plt.tight_layout()
# plt.show()                              #diagramatic schematics

# Visualize the distribution of numerical features with improved readability

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
num_cols = len(numerical_columns)

# # Create the first figure with 8 subplots

fig1, axes1 = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
fig1.suptitle('Distribution of Numerical Features (First 8)', fontsize=16)

for i in range(min(num_cols, 8)):
     sns.histplot(df[numerical_columns[i]], bins=30, kde=True, ax=axes1[i // 4, i % 4], alpha=0.7)
     axes1[i // 4, i % 4].set_title(numerical_columns[i])
     axes1[i // 4, i % 4].set_xlabel(numerical_columns[i])
     axes1[i // 4, i % 4].set_ylabel('Frequency')

# # Hide any unused subplots in the first figure

for j in range(i + 1, 8):
     fig1.delaxes(axes1[j // 4, j % 4])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent clipping
plt.show()

# # Create the second figure for any remaining subplots (next set of up to 8)

if num_cols > 8:
    fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    fig2.suptitle('Distribution of Remaining Numerical Features', fontsize=16)

    for i in range(8, min(num_cols, 16)):
        sns.histplot(df[numerical_columns[i]], bins=30, kde=True, ax=axes2[(i - 8) // 4, (i - 8) % 4], alpha=0.7)
        axes2[(i - 8) // 4, (i - 8) % 4].set_title(numerical_columns[i])
        axes2[(i - 8) // 4, (i - 8) % 4].set_xlabel(numerical_columns[i])
        axes2[(i - 8) // 4, (i - 8) % 4].set_ylabel('Frequency')

# # Hide any unused subplots in the second figure

    for j in range(i + 1 - num_cols):
        fig2.delaxes(axes2[j // 4, j % 4])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()

# Correlation heatmap/ correlation matrix map

# plt.figure(figsize=(12, 10))
# sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap of Numerical Features')
# plt.show() 

# Correlation heatmap with improved readability

plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Step 6: Normalization and Standardization

# Import necessary scaling libraries

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # Identify numerical columns

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# # Method 1: StandardScaler (Z-score normalization)
# # Transforms data to have mean=0 and standard deviation=1

scaler_standard = StandardScaler()
df_standardized = df.copy()
df_standardized[numerical_columns] = scaler_standard.fit_transform(df[numerical_columns])

# # Method 2: MinMaxScaler (Normalization)
# # Scales features to a fixed range, typically between 0 and 1

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

# # Optional: Compare original and transformed data statistics

print("\nOriginal Data Statistics:")
print(df[numerical_columns].describe())

print("\nStandardized Data Statistics:")
print(df_standardized[numerical_columns].describe())

print("\nNormalized Data Statistics:")
print(df_normalized[numerical_columns].describe())

#CONTINUE WITH FEATURE ENGINEERING

# Step 7: Feature Engineering

# Create a copy of the dataframe to work with

# df_engineered = df.copy()

# # 1. Ratio features

# df_engineered['height_to_length_ratio'] = df_engineered['AverageHeight'] / df_engineered['Laenge']
# df_engineered['height_to_breadth_ratio'] = df_engineered['AverageHeight'] / df_engineered['Breite']

# # 2. Difference features

# df_engineered['height_difference'] = df_engineered['MeasureHeight'] - df_engineered['AverageHeight']

# # 3. Aggregate features for lift lead heights

# lift_lead_columns = [col for col in df_engineered.columns if col.startswith('LiftLeadHeight')]
# df_engineered['avg_lift_lead_height'] = df_engineered[lift_lead_columns].mean(axis=1)
# df_engineered['max_lift_lead_height'] = df_engineered[lift_lead_columns].max(axis=1)
# df_engineered['min_lift_lead_height'] = df_engineered[lift_lead_columns].min(axis=1)

# # 4. Composite feature for overall displacement

# df_engineered['total_shift'] = np.sqrt(df_engineered['shiftX']**2 + df_engineered['shiftY']**2)

# # 5. Binary feature for significant rotation

# df_engineered['significant_rotation'] = ((df_engineered['VerkippungX'] > df_engineered['VerkippungX'].mean() + df_engineered['VerkippungX'].std()) | 
#                                          (df_engineered['VerkippungX'] < df_engineered['VerkippungX'].mean() - df_engineered['VerkippungX'].std())).astype(int)

# df_engineered['significant_rotation'] = ((df_engineered['VerkippungY'] > df_engineered['VerkippungY'].mean() + df_engineered['VerkippungY'].std()) | 
#                                          (df_engineered['VerkippungY'] < df_engineered['VerkippungY'].mean() - df_engineered['VerkippungY'].std())).astype(int)

# # # Display the new features

# print(df_engineered[['height_to_length_ratio', 'height_to_breadth_ratio', 'height_difference', 
#                      'avg_lift_lead_height', 'max_lift_lead_height', 'min_lift_lead_height', 
#                      'total_shift', 'significant_rotation']].head())

# # # Correlation analysis of new features

# new_features = ['height_to_length_ratio', 'height_to_breadth_ratio', 'height_difference', 
#                 'avg_lift_lead_height', 'max_lift_lead_height', 'min_lift_lead_height', 
#                 'total_shift', 'significant_rotation']

# plt.figure(figsize=(12, 10))
# sns.heatmap(df_engineered[new_features].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of New Features')
# plt.tight_layout()
# plt.show()

# # List of actually created new features

# new_features = [col for col in df_engineered.columns if col not in df.columns]
# print("Newly created features:")
# print(new_features)

# # # Display the new features

# print(df_engineered[new_features].head())

# # # Correlation analysis of new features

# plt.figure(figsize=(12, 10))
# sns.heatmap(df_engineered[new_features].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of New Features')
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Separate numeric and non-numeric columns

# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns

# print("Numeric columns:", numeric_columns)
# print("Categorical columns:", categorical_columns)

# # Perform correlation analysis on numeric columns only

# correlation_matrix = df[numeric_columns].corr()

# # Visualize the correlation matrix

# plt.figure(figsize=(15, 12))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Feature Correlation Matrix (Numeric Features Only)')
# plt.tight_layout()
# plt.show()

# # Identify highly correlated features

# threshold = 0.9  # You can adjust this threshold
# high_corr_features = set()
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if abs(correlation_matrix.iloc[i, j]) > threshold:
#             colname = correlation_matrix.columns[i]
#             high_corr_features.add(colname)

# print("Highly correlated features:", high_corr_features)

# # Optional: Create a list of features to keep (not highly correlated)

# features_to_keep = [col for col in numeric_columns if col not in high_corr_features]
# print("Features to keep:", features_to_keep)