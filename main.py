import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('sample_data.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Select features for clustering
features = ['Age','program','intake', 'hometown','study_hours', 'monthly_spending', 'sleep_duration', 
           'movie_hours', 'sports_hours', 'book_hours', 'gaming_hours', 
           'social_media_hours']

# Create feature matrix X
X = df[features]
# print(X)

features_list = ['hometown','study_hours', 'sleep_duration', 'movie_hours', 
                     'sports_hours', 'book_hours', 'gaming_hours', 
                     'social_media_hours']
correlation_matrix = df[features_list].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap of Time Features")

# Save the plot as a PNG file
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Exploratory data analysis (EDA)
plt.figure(figsize=(10, 4))
sns.histplot(X['monthly_spending'], bins=50, kde=True, color='skyblue')
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution")

# Save the plot as a PNG file
plt.savefig("income_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

def plot_hist(feature):
    plt.figure(figsize=(9, 4))
    sns.histplot(df[feature], bins=50, kde=True, color='skyblue')
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")
    # Save the plot as a PNG file
    plt.savefig(f"{feature}_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

numeric_variables = ['Age', 'study_hours', 'sleep_duration']

for variable in numeric_variables:
    plot_hist(variable)


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled)


# PCA
# Create a PCA (Principal Component Analysis) object with 2 components and whiten the data
pca = PCA(n_components=2, whiten=True)

# Fit the PCA model to the data
pca.fit(X_scaled)

# Transform the original data into the principal components
data_pca = pca.transform(X_scaled)

# Print the explained variance ratios for each principal component
print("Explained Variance Ratios: ", pca.explained_variance_ratio_)

# Print the total explained variance by the selected components
print("Total Explained Variance: ", sum(pca.explained_variance_ratio_))

# List to store the Within-Cluster-Sum-of-Squares (WCSS) values for different values of k
wcss = []

# Iterate through different values of k (number of clusters)
for k in range(1, 15):
    # Create a KMeans clustering model with the current value of k
    kmeans = KMeans(n_clusters=k)
    
    # Fit the model to the transformed data (principal components)
    kmeans.fit(data_pca)
    
    # Append the WCSS value to the list
    wcss.append(kmeans.inertia_)

# Plot the WCSS values against the number of clusters (k)
plt.plot(range(1, 15), wcss)
plt.xlabel("Number of Clusters (k)")
plt.xticks(range(1, 15, 1))
plt.ylabel("Within-Cluster-Sum-of-Squares (WCSS) Value")
plt.title("Elbow Method for Optimal k Selection")

# Save the plot as a PNG file
plt.savefig("elbow_method_wcss.png", dpi=300, bbox_inches='tight')
plt.show()

kmeans2 = KMeans(n_clusters=3)

clusters = kmeans2.fit_predict(data_pca)

colors = plt.cm.get_cmap('tab10', 3)

for cluster_num in range(3):
    plt.scatter(data_pca[clusters == cluster_num, 0], 
                data_pca[clusters == cluster_num, 1], 
                label=f'{cluster_num + 1}. Cluster', 
                color=colors(cluster_num))

plt.title('Cluster')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

# Save the plot as a PNG file
plt.savefig("clusters_pca_plot.png", dpi=300, bbox_inches='tight')
plt.show()

df['Cluster'] = kmeans2.predict(data_pca)

# Analyze clusters
cluster_means = df.groupby('Cluster')[features].mean()
print("\nCluster Means:")
print(cluster_means)

def assign_cluster_labels(cluster_means):
    # Create fixed mapping for clusters based on known performance
    cluster_labels = {
        0: 'Good',
        2: 'Medium',
        1: 'Poor'
    }
    return cluster_labels
    
# Get cluster labels
cluster_labels = assign_cluster_labels(cluster_means)

# Add performance labels to the dataframe
df['Performance'] = df['Cluster'].map(cluster_labels)

# Print cluster characteristics with labels
print("\nCluster Labels:")
for cluster, label in cluster_labels.items():
    print(f"\n{label} Students (Cluster {cluster}):")
    print('"""""""""""""""""""""""""""""')
    print(cluster_means.loc[cluster])
    print('"""""""""""""""""""""""""""""')
# Visualize clusters using PCA

# Map cluster numbers to their labels
cluster_labels_mapped = [cluster_labels[cluster] for cluster in clusters]

# Update the plotting code to use the mapped labels
colors = plt.cm.get_cmap('tab10', 3)

for cluster_num, label in cluster_labels.items():
    plt.scatter(data_pca[clusters == cluster_num, 0], 
                data_pca[clusters == cluster_num, 1], 
                label=f'{label} (Cluster {cluster_num + 1})', 
                color=colors(cluster_num))

plt.title('Cluster')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

# Save the plot as a PNG file
plt.savefig("mapped_clusters_pca_plot.png", dpi=300, bbox_inches='tight')
plt.show()

standardized_means = pd.DataFrame(
    scaler.transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)


# Rename the index to use performance labels
standardized_means.index = [cluster_labels[i] for i in standardized_means.index]
standardized_means = standardized_means.drop(columns = ['program', 'intake', 'hometown'])

# Create feature importance heatmap with standardized values
plt.figure(figsize=(12, 8))
sns.heatmap(standardized_means, annot=True, cmap='RdYlBu_r', fmt='.2f', center=0)
plt.title('Standardized Feature Patterns by Student Performance')
plt.xlabel('Features')
plt.ylabel('Performance Level')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the heatmap as a PNG file
plt.savefig("standardized_feature_patterns_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Save results
cluster_means.to_csv('cluster_analysis.csv')
df.to_csv('data_with_clusters.csv', index=False)

# Print summary statistics with labels
print("\nNumber of students in each performance category:")
print(df['Performance'].value_counts())

# Calculate and print the silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score: {silhouette_avg:.3f}")