# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:14:15 2022

@author: reco1
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Mall_Customers.csv")
df = data.copy()
print(df)

df.head()
df.describe()
df.info()
df.isnull().sum()

genre = df.iloc[:,1:2].values
age = df.iloc[:,2:3].values
income = df.iloc[:,3:4].values
score = df.iloc[:,4:5].values

# Distributions
# Annual Income
plt.figure(figsize = (8,5))
sns.set(style = "whitegrid")
sns.distplot(income)
plt.title("Distribution of Annual Income (k$)", fontsize = 20)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.show()

# Spending Score
plt.figure(figsize = (8,5))
sns.set(style = "whitegrid")
sns.distplot(score)
plt.title("Distribution of Spending Score (1-100)", fontsize = 20)
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Count")
plt.show()

# Age
plt.figure(figsize = (8,5))
sns.set(style = "whitegrid")
sns.distplot(age)
plt.title("Distribution of Age", fontsize = 20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Genre
genres = df.Genre.value_counts()
plt.figure(figsize = (8,5))
sns.set(style = "darkgrid")
sns.barplot(x = genres.index, y = genres.values)
plt.title("Distribution of Genre", fontsize = 20)
plt.show()

#Categorical -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
genre[:,0] = le.fit_transform(genre[:,0])
print(genre)

ohe = preprocessing.OneHotEncoder()
genre = ohe.fit_transform(genre).toarray()
print(genre)

new_genre = pd.DataFrame(data = genre, index = range(200), columns=("Female","Male"))
print(new_genre)

df = pd.concat([df, new_genre], axis = 1)
print(df)

del df["Genre"]
print(df)


#Clustering
from sklearn.cluster import KMeans

x = data.iloc[:,[3,4]].values
wcss = []
kume_sayisi_listesi = range(1,11)
for i in kume_sayisi_listesi:
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Elbow Curve
plt.plot(kume_sayisi_listesi, wcss)
plt.title("Küme Sayısı için Dirsek Yöntemi", fontsize = 15)
plt.xlabel("Küme Sayısı")
plt.ylabel("WCSS")
plt.xticks(np.arange(1,11,1))
plt.show()

#Annual Income vs Spending Score
X = df[["Annual Income (k$)","Spending Score (1-100)"]]
plt.figure(figsize = (12,6))
sns.scatterplot(x = df["Annual Income (k$)"], y = df["Spending Score (1-100)"], data = X, s = 100)
plt.title("Annual Income (k$) vs Spending Score (1-100)", fontsize = 20)
plt.xlabel("Annual Income (k$)", fontsize = 15)
plt.ylabel("Spending Score (1-100)", fontsize = 15)
plt.show()


#Belirlenen küme sayısına göre kümeleme yapmak
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize = (12,5))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Küme 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Küme 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Küme 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Küme 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Küme 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Küme Merkezleri')
sns.set(style = "whitegrid")
plt.title('Müşteri Segmentasyonu')
plt.xlabel('Yıllık Gelir')
plt.ylabel('Harcama Skoru (1-100)')
plt.legend()
plt.show()
