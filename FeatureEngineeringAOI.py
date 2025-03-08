import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AOI_Data.csv' 

# Read the CSV file

df = pd.read_csv(file_path)

# Visualize the distribution of numerical features

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Step 7: Feature Engineering

# Create a copy of the dataframe to work with

df_engineered = df.copy()

# 1. Ratio features

df_engineered['height_to_length_ratio'] = df_engineered['AverageHeight'] / df_engineered['Laenge']
df_engineered['height_to_breadth_ratio'] = df_engineered['AverageHeight'] / df_engineered['Breite']

# 2. Difference features

df_engineered['height_difference'] = df_engineered['MeasureHeight'] - df_engineered['AverageHeight']

# 3. Aggregate features for lift lead heights

lift_lead_columns = [col for col in df_engineered.columns if col.startswith('LiftLeadHeight')]
df_engineered['avg_lift_lead_height'] = df_engineered[lift_lead_columns].mean(axis=1)
df_engineered['max_lift_lead_height'] = df_engineered[lift_lead_columns].max(axis=1)
df_engineered['min_lift_lead_height'] = df_engineered[lift_lead_columns].min(axis=1)

# 4. Composite feature for overall displacement

df_engineered['total_shift'] = np.sqrt(df_engineered['shiftX']**2 + df_engineered['shiftY']**2)

# 5. Binary feature for significant rotation

df_engineered['significant_rotation'] = ((df_engineered['VerkippungX'] > df_engineered['VerkippungX'].mean() + df_engineered['VerkippungX'].std()) | 
                                         (df_engineered['VerkippungX'] < df_engineered['VerkippungX'].mean() - df_engineered['VerkippungX'].std())).astype(int)

df_engineered['significant_rotation'] = ((df_engineered['VerkippungY'] > df_engineered['VerkippungY'].mean() + df_engineered['VerkippungY'].std()) | 
                                         (df_engineered['VerkippungY'] < df_engineered['VerkippungY'].mean() - df_engineered['VerkippungY'].std())).astype(int)

# # Display the new features

# print(df_engineered[['height_to_length_ratio', 'height_to_breadth_ratio', 'height_difference', 
#                      'avg_lift_lead_height', 'max_lift_lead_height', 'min_lift_lead_height', 
#                      'total_shift', 'significant_rotation']].head())

# # Correlation analysis of new features

# new_features = ['height_to_length_ratio', 'height_to_breadth_ratio', 'height_difference', 
#                 'avg_lift_lead_height', 'max_lift_lead_height', 'min_lift_lead_height', 
#                 'total_shift', 'significant_rotation']

# plt.figure(figsize=(12, 10))
# sns.heatmap(df_engineered[new_features].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of New Features')
# plt.tight_layout()
# plt.show()

# List of actually created new features
# new_features = [col for col in df_engineered.columns if col not in df.columns]

# print("Newly created features:")
# print(new_features)

# # Display the new features
# print(df_engineered[new_features].head())

# # Correlation analysis of new features
# plt.figure(figsize=(12, 10))
# sns.heatmap(df_engineered[new_features].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of New Features')
# plt.tight_layout()
# plt.show()

# Identify highly correlated features
correlation_matrix = df.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Select features with correlation less than a threshold
threshold = 0.8
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)

print("Highly correlated features:", high_corr_features)
