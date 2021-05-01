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

#FIND RECOMMENDATIONS ------------------------------------------------------
    
POI_ID = "P10-1142"
#POI_ID = "P12-1041"
POI_INDEX = papers[POI_ID].pid


#Making Training Data
trainingData = data.drop(POI_INDEX, axis=0, inplace=False)
trainingData.drop(POI_INDEX, axis=1, inplace=True)
#Shape(24627,25627)

#Making Testing Data
testingData = data.iloc[POI_INDEX]
testingData.drop(POI_INDEX, inplace=True)
#shape (1,24627)

#%%

    #Training -----------------------
kMeans = KMeans(n_clusters=15, max_iter=100, random_state=0)
AllotedClustersTraining = kMeans.fit_predict(trainingData)

#%%

    #Testing -----------------------
testingData = testingData.values.reshape(1,-1)
AllotedCluster = kMeans.predict(testingData)[0]

#%%

# Finding Predictions
clusterArray = []
nonclusterArray = []
for i in range(len(AllotedClustersTraining)):
    if(AllotedClustersTraining[i]==AllotedCluster):
        clusterArray.append(i)
    else:
        nonclusterArray.append(i)


distanceMeasure = 'euclidean'
distanceArray = {}
for i in range(len(clusterArray)):
    trainingclusterCitations = trainingData.iloc[clusterArray[i]].values.reshape(1,-1)
    d = distance.cdist(trainingclusterCitations, testingData, distanceMeasure)
    distanceArray[clusterArray[i]] = d[0][0]
    
dict(sorted(distanceArray.items(), key=lambda item: item[1]))

#%%

#Finding Recommendations 
print("Index of paper of Interest- ", POI_INDEX)
print("Papers Recommended for Paper ID- ", POI_ID)
print("Title- " , papers[POI_ID].title)

#total citations in POI
citationsOriginal = data.iloc[POI_INDEX].values
citationsOriginal = np.delete(citationsOriginal, POI_INDEX)
totalOriginalCitations = np.count_nonzero(citationsOriginal == 1) #22
c1 = np.where(citationsOriginal == 1)[0]

#%%
      
#Find citations which are common with POI and papers ourside our clusters
#this means they were true but model marked them as negative
FalseNegative = 0
for i in range(len(nonclusterArray)):
    trainingNonclusterCitations = trainingData.iloc[nonclusterArray[i]].values
    
    c2 = np.where(trainingNonclusterCitations == 1)[0]
    #print(len(c2))
    c = np.sum(c1 == c2)
    
#    if(c>0):
#        print(c," " ,c1," ", c2)
#        print(nonclusterArray[i])
    FalseNegative += (c/totalOriginalCitations)
    
    
#%%
    
#Find citations which are not common with POI and papers ourside our clusters
#this means they were false and model marked them as negative
TrueNegative = 0
for i in range(len(nonclusterArray)):
    trainingNonclusterCitations = trainingData.iloc[nonclusterArray[i]].values

    c = np.sum(citationsOriginal != trainingNonclusterCitations)
#    if(c>1):
#        print(c)
#        print(nonclusterArray[i])
    TrueNegative += (c/len(citationsOriginal))
   
    
#%%
 
##Find citations which are not common with POI and papers in our clusters
##this means they were false and model marked them as positive
#FalsePositive = 0
#TruePositive = 0
#for i in range(len(clusterArray)):
#    trainingclusterCitations = trainingData.iloc[clusterArray[i]].values
#
#    c2 = np.where(trainingclusterCitations == 1)[0]
#    
#    common =0 
#    for i in range(len(citationsOriginal)):
#        if(citationsOriginal[i]==1 and citationsOriginal[i] == trainingclusterCitations[i]):
#            common += 1
#
#    FalsePositive += ((len(c2) - common)/len(c2))
#    TruePositive += (common/len(c1))


#%%
    

k = 15
recallArray = []
precisionArray = []
accuracyArray=[]

for i in range(1,k+1):
    topKPapers = clusterArray[:i]
    print(topKPapers)
    
    FalsePositive = 0
    TruePositive = 0
    for i in range(len(topKPapers)):
        trainingclusterCitations = trainingData.iloc[topKPapers[i]].values
        c2 = np.where(trainingclusterCitations == 1)[0]
        
        common =0 
        for i in range(len(citationsOriginal)):
            if(citationsOriginal[i]==1 and citationsOriginal[i] == trainingclusterCitations[i]):
                common += 1
    
        FalsePositive += ((len(c2) - common)/len(c2))
        TruePositive += (common/len(c1))
    
    recall = TruePositive / (TruePositive + FalseNegative)
    precision = TruePositive / (TruePositive + FalsePositive)
    accuracy = (TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative)
    
    recallArray.append(recall)
    precisionArray.append(precision)
    accuracyArray.append(accuracy)


#%%
#from scipy import spatial
#
#citationsOriginal = data.iloc[POI_INDEX]
#totalOriginalCitations = 0
#for i in range(len(citationsOriginal)):
#    if(citationsOriginal[i]==1):
#        totalOriginalCitations += 1
#        
#recallArray = []
#precisionArray = []
#for paper in recommendedPapers:
#    citationsPredicted = data.iloc[paper.pid]
#    common = 0
#    
#    for i in range(len(citationsOriginal)):
#        if(citationsOriginal[i]==1 and citationsOriginal[i] == citationsPredicted[i]):
#            common += 1
#    totalPredictedCitations = 0
#    for i in range(len(citationsPredicted)):
#        if(citationsPredicted[i]==1):
#            totalPredictedCitations += 1
#            
#    recall = common / totalOriginalCitations
#    precision = common / totalPredictedCitations
#    #result = 1 - spatial.distance.cosine(citationsOriginal, citationsPredicted)
#    print("Recall " , recall, " Precision " , precision)
#    recallArray.append(recall)
#    precisionArray.append(precision)
    
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
plt.show()   

#%%

  
#plt.plot(Xaxis, precisionArray)
#plt.title('Precision Graph')
#plt.xlabel('Recommend List of Papers')
#plt.ylabel('Precision Scores')
#plt.show()    
    