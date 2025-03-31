import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA


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




df = pd.read_csv('../AstroDataset/star_classification.csv')

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation = X.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Between Photometric Bands')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# PCA for visualization and dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize data in 2D PCA space with class colors
plt.figure(figsize=(12, 10))
classes = df['class'].unique()
colors = ['b', 'r', 'g']  # blue for GALAXY, red for QSO, green for STAR
for i, cls in enumerate(classes):
    plt.scatter(X_pca[df['class'] == cls, 0], 
                X_pca[df['class'] == cls, 1],
                c=colors[i], label=cls, alpha=0.5)
plt.title('PCA of Astronomical Objects')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('pca_visualization.png')
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method to find optimal K
# Just use inertia (elbow method) on the full dataset
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"K={k}, Inertia={kmeans.inertia_:.2f}")

# Plot just the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('optimal_k_inertia.png')
plt.show()