# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:39:53 2021

@author: Sezal
"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from scipy.spatial import distance
import math
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

class paper:
  def __init__(self,pid, ID, title, year):
    self.pid = pid
    self.ID = ID
    self.title = title
    self.year = year
    
papers={}

with open("datasets_inUse/paper_ids.txt","r", encoding="utf8") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # making the entire title sentence
        title=' '.join(l[1:len(l)-1])
        # paper id pid is increasing values of 1 with eveyr loop
        papers[l[0]]=paper(pid, l[0], title, l[-1])
        pid+=1


#Number of papers in total
nop=len(papers)

""""Paper Citation matrix"""

def paper_citation_matrix():
    with open("datasets_inUse/paper-citation-network-nonself.txt",'r') as file:
        matrix=np.zeros((nop,nop))
        for i in tqdm(file.readlines()):
            l=i.split()
            #print(papers[l[0]].pid," " , papers[l[2]].pid," -------------------")
            matrix[papers[l[0]].pid,papers[l[2]].pid]=1
    return matrix






matrix=paper_citation_matrix()
# np.save("matrix", matrix)
print(matrix.shape)
data=pd.DataFrame(matrix)
print(data.shape)

#Pickling the Citation Matrix
#pickling_on = open("CitationMatrix.pickle","wb")
#pickle.dump(data, pickling_on,protocol=4)
#pickling_on.close()

#%%

#K means clustering from SCRATCH 

def Kmeans(movieRatings, n_iter, K, distanceMeasure):  
    
    #Initialization Stage
    if isinstance(movieRatings, pd.DataFrame):
        X = movieRatings.values
    idx = np.random.choice(len(X), K, replace=False)
    centroids = X[idx, :]
    
    #Get Min values for all users from centriods and get their initial clusters
    P = np.argmin(distance.cdist(X, centroids, distanceMeasure),axis=1)
    #shape of P is (X.shape[0], 1)
    
    #Looping Stage
    for _ in range(n_iter):
            centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(K)])
            tmp = np.argmin(distance.cdist(X, centroids, distanceMeasure),axis=1)
            if np.array_equal(P,tmp):break
            P = tmp       
    #print(P[0:50])
    
    finalCentroid=[0 for i in range(K)]
    finalDistance=[math.inf for i in range(K)]
    
    
    for i in range(len(P)):    
        initalrating= movieRatings.iloc[i].values.reshape(1,1664)
        currentCentroid= centroids[P[i]].reshape(1,1664)
        
    
        dist = distance.cdist(initalrating, currentCentroid, distanceMeasure)
        #print(dist)
        if(dist<finalDistance[P[i]]):
            #print(dist, " " , i, " " , P[i])
            finalDistance[P[i]]=dist
            finalCentroid[P[i]]=i    
            
    return finalCentroid  , P

#%% --------------------------------------------------------------------
    #CALCULATE MAE 
    #This is just for testing purposes of the ALGO
# ---------------------------------------------------------DONT RUN THIS CELL

cv = KFold(5, shuffle=False, random_state=None)
foldnumber=1

for train_index, test_index in cv.split(data):
    print("Fold ",foldnumber)
    foldnumber += 1
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    print(train_index, " " , test_index)
    print(train.shape, " TRAIN " , test.shape, " TEST ")
    
    #Train the dataset to get cluster centroids
    #centroids, clusters= Kmeans(train, n_iter, n_clusters, distanceMeasure)
    km = KMeans(n_clusters=10, max_iter=100, random_state=0).fit(train)
    #Training -----------------------
    #km = TimeSeriesKMeans(n_clusters=10, metric="euclidean", max_iter=10,random_state=0).fit(train)
    print(km.cluster_centers_.shape)
    centroids = km.cluster_centers_
    
    #Testing -----------------------
    predicted = km.predict(test)
    #print(km.cluster_centers_)
    
    #MAE Calculation -------------------
    maeFold = 0
    for row in range(len(predicted)):
        clusterNumberPredicted = predicted[row]
        predictedCitations = centroids[clusterNumberPredicted]
        
        originalCitations = test.iloc[row]
        
        mae = mean_absolute_error(originalCitations, predictedCitations)
        maeFold += mae
    
    print("MAe value for fold ", (foldnumber-1) , " is ", maeFold)
    
    
#%%

#FIND RECOMMENDATIONS ------------------------------------------------------
    
POI_ID = "P12-1041"
POI_INDEX = papers[POI_ID].pid


#Making Training Data
trainingData = data.drop(POI_INDEX, axis=0, inplace=False)
trainingData.drop(POI_INDEX, axis=1, inplace=True)

#Making Testing Data
testingData = data.iloc[POI_INDEX]
testingData.drop(POI_INDEX, inplace=True)

    #Training -----------------------
kMeans = KMeans(n_clusters=10, max_iter=100, random_state=0)
AllotedClustersTraining = kMeans.fit_predict(trainingData)

    #Testing -----------------------
testingData = testingData.values.reshape(1,-1)
AllotedCluster = kMeans.predict(testingData)


# Finding Predictions
clusterArray = []
for i in range(len(AllotedClustersTraining)):
    if(AllotedClustersTraining[i]==AllotedCluster):
        clusterArray.append(i)


distanceMeasure = 'euclidean'
distanceArray = {}
for i in range(len(clusterArray)):
    trainingclusterCitations = trainingData.iloc[clusterArray[i]].values.reshape(1,-1)
    d = distance.cdist(trainingclusterCitations, testingData, distanceMeasure)
    distanceArray[clusterArray[i]] = d[0][0]
    
dict(sorted(distanceArray.items(), key=lambda item: item[1]))

#Finding Recommendations 
print("Index of paper of Interest- ", POI_INDEX)
print("Papers Recommended for Paper ID- ", POI_ID)
print("Title- " , papers[POI_ID].title)
topKPapers = 5
for i in range(topKPapers):
    pid = list(distanceArray.keys())[i]
    for j in papers:
        if(papers[j].pid==pid):
            print(i+1, ". ", papers[j].title , " " , j)
