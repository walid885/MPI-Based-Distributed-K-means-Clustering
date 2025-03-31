import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../AstroDataset/star_classification.csv')

# Replace -9999 values with NaN (Not a Number)
df.replace(-9999.0, np.nan, inplace=True)

# See how many NaN values we have now
print("Missing values after replacement:")
print(df.isnull().sum())

# Option 1: Drop rows with missing values
# df_clean = df.dropna()

# Option 2: Impute missing values (with median)
for col in ['u', 'g', 'r', 'i', 'z']:
    df[col].fillna(df[col].median(), inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum())

# Check class distribution
class_counts = df['class'].value_counts()
print("\nClass Distribution:")
print(class_counts)

# Visualize class distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Distribution of Astronomical Object Classes')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()


# Select features for clustering
features = ['u', 'g', 'r', 'i', 'z']  # Photometric bands
X = df[features]

# Normalize the features (very important for K-means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# See summary statistics of scaled features
print("\nScaled Features Statistics:")
print(X_scaled_df.describe())

