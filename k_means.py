import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Load the data
df = pd.read_csv('heart.csv')
print(df.head())

# Check for null values and remove them
print(df.isnull().sum())
data = df.dropna()

# Features for clustering
X = df[['age', 'trtbps', 'chol']]

print(X.head())
print(X.tail())
#Using elbow method 
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['age','trtbps','chol']])
    sse.append(km.inertia_)
plt.xlabel('k')
plt.ylabel('sqaured Error')
plt.plot(k_rng,sse)
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)  
kmeans.fit(X)

# Predict the clusters
clusters = kmeans.predict(X)
df['Cluster'] = clusters  

print(f'Cluster centers:{kmeans.cluster_centers_}')
print(f'Cluster labels:{df['Cluster'].value_counts()}')

# Plotting the clusters based on 'age' and 'chol'
plt.scatter(df['age'], df['chol'], c=df['Cluster'])
plt.xlabel('Age')
plt.ylabel('Cholesterol (chol)')
plt.title('Age vs Cholesterol Clustering')
plt.show()
