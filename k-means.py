#Importing the libreries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


##Working with our dataset
df =  pd.read_csv("ObesityDataSet.csv")
#print(df.head())
#print(df.columns)

#After checking our dataset we need to convert non-numerical variables to numerical in order to work with k-means clustering
#We transform binary columns in 1s and 0s
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

#We label categorical data
categorical_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df.astype({col: int for col in df.select_dtypes('bool').columns})
df = df.drop(columns=['NObeyesdad'])

#We normalize numerical values
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()  
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

##Lets apply the elbow method to find the # of clusters
# Define the range of clusters to test (from 1 to 10)
K_range = range(1, 11)
wcss = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit the model to the normalized data
    kmeans.fit(df)
    
    # Store the WCSS value (inertia)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph (The elbow doesn't show a specific k value to use)
"""plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method to Determine Optimal k')
plt.show()"""

#Lets use teh silhouette method 
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    print(f'K={k}, Silhouette Score={score}')

#The method shows that K=8 would be the most appropriate value, however, it doesn't show a big difference between other K values