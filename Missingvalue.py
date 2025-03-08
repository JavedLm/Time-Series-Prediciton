#Implement this code after line 113 to find missing valuesin the dataset

#AOI doesn't contain any missing values and therefore this part is not required

# Step 5: Handle missing values

# Check for missing values
# print("Missing values in each column:")
# print(df.isnull().sum())

# # Calculate percentage of missing values
# print("\nPercentage of missing values:")
# print(df.isnull().sum() / len(df) * 100)

# # Decide on strategy for handling missing values
# # For this example, we'll use simple imputation methods

# # For numerical columns, fill with median
# numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# # For categorical columns (if any), fill with mode
# categorical_columns = df.select_dtypes(include=['object']).columns
# df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# # Verify that missing values have been handled
# print("\nMissing values after imputation:")
# print(df.isnull().sum())

# # Optional: Drop rows with any remaining missing values
# # df = df.dropna()

# # Display the first few rows of the updated dataset
# print("\nUpdated dataset:")
# print(df.head())