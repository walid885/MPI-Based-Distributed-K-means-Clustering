import pandas as pd
"""
the first interaction with the dataset, in order 
to get an idea about the quality of the data set , as well , 
how well is it established ! 
"""
# Assuming your dataset is in a CSV format
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('../AstroDataset/star_classification.csv')

# Display the first 10 rows
print(df.head(10))

# Basic dataset information
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Number of features: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")

# Data types of columns
print("\nData Types:")
print(df.dtypes)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Check for unique values in each column (for categorical features)
print("\nUnique Values in Each Column:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {df[col].nunique()} unique values")