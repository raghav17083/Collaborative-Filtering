# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:04:31 2021

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
from sklearn import preprocessing
from Kernel_K_Means import Kernel_K_Means


class paper:
   def __init__(self,pid, ID, title, year):
     self.pid = pid
     self.ID = ID
     self.title = title
     self.year = year
    
papers={}
pid_PaperID = {}
with open("datasets_inUse/paper_ids.txt","r", encoding="utf8") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # making the entire title sentence
        title=' '.join(l[1:len(l)-1])
        # paper id pid is increasing values of 1 with eveyr loop
        papers[l[0]]=paper(pid, l[0], title, l[-1])
        pid_PaperID[pid] = l[0]
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
#%%

#Making Testing Data
POI_ID = "P10-1142"
#POI_ID = "P12-1041"
POI_INDEX = papers[POI_ID].pid

similarities =[0] * matrix.shape[0]


#%%

citationsOriginal = matrix[POI_INDEX,:]
citationsTestingData = np.where(citationsOriginal == 1)[0]
    
for i in range(matrix.shape[0]):
  # for j in range(matrix.shape[0]):
    # print(i)
    if(i==POI_INDEX):
       similarities[i]= -1

    else:
      trainingData = matrix[i,:]
      citationsTrainingData = np.where(trainingData == 1)[0]
      co_citations = list(set(citationsTrainingData) & set(citationsTestingData))
      if(len(co_citations)==0):
        similarities[i]=0
      else :
        sim = 0
        for i in co_citations:
            temp = matrix[:,i]
            total_links =  len(np.where(temp == 1)[0])
            sim = sim + (1/(total_links-1))
            
        similarities[i] = sim
        
#%%

CitationsSelectedPapers = np.argsort(similarities)[::-1][:len(similarities)]
#print(CitationsSelectedPapers)
        
#%%
print("Papers Recommended for Paper - " ,POI_ID)
topKPapers = 5
for i in range(0, topKPapers):
    pid = CitationsSelectedPapers[i]
    print(i+1, ". ", papers[pid_PaperID[pid]].title , " -", papers[pid_PaperID[pid]].ID)

#%%


recallArray = []
precisionArray = []
accuracyArray=[]

for topKPapers in range(1,7):
    print("-------------------", topKPapers)
    pidSelectedPapers = CitationsSelectedPapers[:topKPapers]
    pidNonSelectedPapers = CitationsSelectedPapers[topKPapers:]
    
    #total citations in POI
    citationsOriginal = data.iloc[POI_INDEX].values
    citationsOriginal = np.delete(citationsOriginal, POI_INDEX)
    totalOriginalCitations = np.count_nonzero(citationsOriginal == 1)
    c1 = np.where(citationsOriginal == 1)[0]
    
    
    #Find citations which are common with POI and papers ourside our clusters
    #this means they were true but model marked themas negative
    FalseNegative = 0
    for i in range(len(pidNonSelectedPapers)):
        trainingNonclusterCitations = data.iloc[pidNonSelectedPapers[i]].values
        
        c2 = np.where(trainingNonclusterCitations == 1)[0]
        #print(len(c2))
        c = np.sum(c1 == c2)
        FalseNegative += (c/totalOriginalCitations)
    
    #Find citations which are not common with POI and papers ourside our clusters
    #this means they were false and model marked them as negative
    TrueNegative = 0
    for i in range(len(pidNonSelectedPapers)):
        trainingNonclusterCitations = data.iloc[pidNonSelectedPapers[i]].values
    
        c = np.sum(citationsOriginal != trainingNonclusterCitations)
        TrueNegative += (c/len(citationsOriginal))
    
    
    #Find citations which are in common with POI and papers in our clusters
    #this means they were true and model marked them as positive
        
    #Find citations which are not common with POI and papers in our clusters
    #this means they were false and model marked them as negative
    FalsePositive = 0
    TruePositive = 0
    for i in range(len(pidSelectedPapers)):
        trainingclusterCitations = data.iloc[pidSelectedPapers[i]].values
        c2 = np.where(trainingclusterCitations == 1)[0]
        
        common =0 
        for i in range(len(citationsOriginal)):
            if(citationsOriginal[i]==1 and citationsOriginal[i] == trainingclusterCitations[i]):
                common += 1
                
        if(len(c2)!=0 and len(c1)!=0):
            FalsePositive += ((len(c2) - common)/len(c2))
            TruePositive += (common/len(c1))
    
    #RECALL PRECISION ACCURACY
    print(TruePositive, " " ,TrueNegative )
    print(FalsePositive, " ", FalseNegative)
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
   
Xaxis = [i for i in range(6)]
  
plt.plot(Xaxis, PlotRecall, c='red', label='Recall')
plt.plot(Xaxis, PlotPrecision, c='blue', label ='Precision')
plt.title('Recall and Precision Graph')
plt.xlabel('List of top K Recommended Papers')
plt.ylabel('Cummulative Average Scores')
plt.legend()
plt.show()   

#%%
print("Recall", PlotRecall)
print("Precision", PlotPrecision)



