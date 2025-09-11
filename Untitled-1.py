# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv("Mall_Customers.csv")  # Update path if needed
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize initial data
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c='gray', s=50)
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Customers: Income vs Spending Score')
plt.show()

# Determine optimal number of clusters using Elbow Method
inertia = []
K = range(1,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Fit K-Means with chosen number of clusters (example: k=5)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(8,5))
for i in range(k_optimal):
    plt.scatter(X_scaled[y_kmeans==i,0], X_scaled[y_kmeans==i,1], label=f'Cluster {i+1}', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Customer Segments (K-Means Clustering)')
plt.legend()
plt.show()

# Optional: Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
print(f'Silhouette Score: {score:.2f}')
