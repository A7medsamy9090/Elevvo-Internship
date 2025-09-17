import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- KMeans ----------
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['KMeans'], cmap='rainbow', s=40)
plt.title("KMeans Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

print("\nKMeans cluster averages:")
print(df.groupby('KMeans')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

# ---------- DBSCAN ----------
db = DBSCAN(eps=0.5, min_samples=5)  # adjust eps if needed
df['DBSCAN'] = db.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['DBSCAN'], cmap='plasma', s=40)
plt.title("DBSCAN Clusters (noise = -1)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

print("\nDBSCAN cluster averages (ignoring noise):")
print(df[df['DBSCAN']!=-1].groupby('DBSCAN')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
