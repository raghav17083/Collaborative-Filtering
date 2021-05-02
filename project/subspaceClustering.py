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

#affinity = ['symmetrize','nearest_neighbors']
affinity = 'symmetrize'
ssc = SparseSubspaceClusteringOMP(n_clusters=10, affinity=affinity)
clusterTrain = ssc.fit(trainingData)
AllotedClustersTraining=clusterTrain.labels_

AllotedCluster = ssc.fit_predict(testingData)
from scipy import stats
from collections import Counter

m = stats.mode(AllotedCluster)
print(m)
Alloted = 0
Counter(AllotedCluster)

#%%

# Finding Predictions
clusterArray = []
nonclusterArray = []
for i in range(len(AllotedClustersTraining)):
    if(AllotedClustersTraining[i]==Alloted):
        clusterArray.append(i)
    else:
        nonclusterArray.append(i)
        
#%%
testingData = testingData.T
print(testingData.shape)

#%%
#distanceMeasure = 'euclidean'
#distanceMeasure = 'cosine'
distanceMeasure = 'jaccard'
distanceArray = {}
for i in range(len(clusterArray)):
    trainingclusterCitations = trainingData.iloc[clusterArray[i]].values.reshape(1,-1)    
    d = distance.cdist(trainingclusterCitations, testingData, distanceMeasure)
    distanceArray[clusterArray[i]] = d[0][0]
    
    
sorted_dict = [(value, key) for (key, value) in distanceArray.items()]
sorted_dict.sort(reverse=True)
#%%       

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


#%%
#total citations in POI
citationsOriginal = data.iloc[POI_INDEX].values
citationsOriginal = np.delete(citationsOriginal, POI_INDEX)
totalOriginalCitations = np.count_nonzero(citationsOriginal == 1) #22
c1 = np.where(citationsOriginal == 1)[0]
print(len(c1))
print(totalOriginalCitations)
#%%

#%%
      
#Find citations which are common with POI and papers ourside our clusters
#this means they were true but model marked them as negative
FalseNegative = 0
for i in range(len(nonclusterArray)):
    trainingNonclusterCitations = data.iloc[nonclusterArray[i]].values
    
    c2 = np.where(trainingNonclusterCitations == 1)[0]
    #print(len(c2))
    c = np.sum(c1 == c2)
    
#    if(c>0):
#        print(c," " ,c1," ", c2)
#        print(nonclusterArray[i])
    FalseNegative += (c/totalOriginalCitations)
print('FalseNegative', FalseNegative)    
#%%
    
#Find citations which are not common with POI and papers ourside our clusters
#this means they were false and model marked them as negative
TrueNegative = 0
for i in range(len(nonclusterArray)):
    trainingNonclusterCitations = data.iloc[nonclusterArray[i]].values

    c = np.sum(citationsOriginal != trainingNonclusterCitations)
#    if(c>1):
#        print(c)
#        print(nonclusterArray[i])
    TrueNegative += (c/len(citationsOriginal))
print('TrueNegative', TrueNegative)


#%%
k = 15
recallArray = []
precisionArray = []
accuracyArray=[]

for i in range(1,k+1):
    topKPapers = [key for (value,key) in sorted_dict[:i]]
    #print(topKPapers)
    
    FalsePositive = 0
    TruePositive = 0
    for i in range(len(topKPapers)):
        trainingclusterCitations = data.iloc[topKPapers[i]].values
        c2 = np.where(trainingclusterCitations == 1)[0]
        
        common =0 
        for i in range(len(citationsOriginal)):
            if(citationsOriginal[i]==1 and citationsOriginal[i] == trainingclusterCitations[i]):
                common += 1
        if(len(c1)!=0 and len(c2)!=0):
            FalsePositive += ((len(c2) - common)/len(c2))
            TruePositive += (common/len(c1))
    print('Common', common)   
    print('FalsePositive', FalsePositive)
    print('TruePositive', TruePositive)
    print(topKPapers)
    if(TruePositive!=0 or FalsePositive!=0):
        recall = TruePositive / (TruePositive + FalseNegative)
        precision = TruePositive / (TruePositive + FalsePositive)
    accuracy = (TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative)
    
    recallArray.append(recall)
    precisionArray.append(precision)
    accuracyArray.append(accuracy)

    
#%%
accuracyArray = np.cumsum(accuracyArray)
PlotAccuracy = [accuracyArray[i]/(i+1) for i in range(len(accuracyArray))]

recallArray = np.cumsum(recallArray)
PlotRecall = [recallArray[i]/(i+1) for i in range(len(recallArray))]

precisionArray = np.cumsum(precisionArray)
PlotPrecision = [precisionArray[i]/(i+1) for i in range(len(precisionArray))]


#%%
import matplotlib.pyplot as plt
   
Xaxis = [i for i in range(15)]
  
plt.plot(Xaxis, PlotRecall, c='red', label='Recall')
plt.plot(Xaxis, PlotPrecision, c='blue', label ='Precision')
plt.title('Recall and Precision Graph')
plt.xlabel('List of top K Recommended Papers')
plt.ylabel('Cummulative Average Scores')
plt.legend()
name = 'Subspace_' + distanceMeasure + POI_ID + '.png'
plt.savefig(name)
plt.show()   
            
            
