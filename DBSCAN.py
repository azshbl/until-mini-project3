import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

df=pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.dtypes)
df.head()

print("Missing values per column:")
print(df.isnull().sum())


df.duplicated().sum()

df= df.drop(columns=["CustomerID"])

label_ecoder=LabelEncoder()
df["Gender_encoded"] = label_ecoder.fit_transform(df["Gender"])

df


scaled=StandardScaler()
scaled_feu=scaled.fit_transform(df[['Age','Annual Income (k$)','Spending Score (1-100)']])
scaled_df = pd.DataFrame(scaled_feu, columns=["Age_scaled","Income(k$)_scaled","Spending_scaled"])

scaled_df

#DBSCAN

X_scaled = scaled_df[["Income(k$)_scaled", "Spending_scaled"]].values

from sklearn.neighbors import NearestNeighbors

min_samples = 5

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, min_samples-1])

plt.figure(figsize=(8,5))
plt.plot(distances)
plt.title("K-distance Graph")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}th Nearest Neighbor Distance")
plt.grid(True)
plt.show()


db = DBSCAN(eps=0.46, min_samples=5)
clusters_db= db.fit_predict(X_scaled)
l=len(np.unique(clusters_db))
print(f"Number of clusters: {l}")
ll=db.labels_
print(ll)


cluster_c = pd.Series(ll).value_counts()
cluster_c


mask = clusters_db != -1  

if len(set(clusters_db[mask])) > 1:
    score = silhouette_score(X_scaled[mask], clusters_db[mask])
    print(f"Silhouette Score (DBSCAN): {score:.3f}")



plt.figure(figsize=(7,5))

df['DBSCAN_Cluster'] = clusters_db
sns.countplot(
    x='DBSCAN_Cluster',
    data=df,
    order=sorted(df['DBSCAN_Cluster'].unique())
)

plt.title("Number of Customers in Each DBSCAN Cluster")

plt.show()



plt.scatter(scaled_feu[:,1],scaled_feu[:,2],c=clusters_db, cmap="viridis")
plt.title("DBSCAN Clustering")
plt.show()





