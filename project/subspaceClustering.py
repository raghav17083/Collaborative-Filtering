# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:11:55 2021

@author: Sezal
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from scipy.spatial import distance
import math
from selfrepresentation import SparseSubspaceClusteringOMP

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
    return pd.DataFrame(matrix)






data=paper_citation_matrix()
print(data.shape)

#Pickling the Citation Matrix
#pickling_on = open("CitationMatrix.pickle","wb")
#pickle.dump(data, pickling_on,protocol=4)
#pickling_on.close()



    
def calculate(trainingData, testingData, distanceMeasure, n_clusters, affinity):  

        ssc = SparseSubspaceClusteringOMP(n_clusters, affinity=affinity)
        clusterTrain = ssc.fit(trainingData)
        TrainingClusters=clusterTrain.labels_
        clusteringTest = ssc.fit_predict(testingData)

        return TrainingClusters,clusteringTest
        



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

affinity = ['symmetrize','nearest_neighbors']

for a in affinity:
    AllotedClustersTraining,AllotedCluster = calculate(trainingData, testingData, distanceMeasure='euclidean', n_clusters=10,affinity=a)



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

