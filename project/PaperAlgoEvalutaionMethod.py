# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:29:47 2021

@author: Sezal
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from collections import Counter

from paper_class import paper

# class paper:
#   def __init__(self,pid, ID, title, year):
#     self.pid = pid
#     self.ID = ID
#     self.title = title
#     self.year = year
    
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
#Paper of Interest 
POI_ID = "P12-1041"
POI_INDEX = papers[POI_ID].pid
print("Index of paper of Interest- ", POI_INDEX)

#All papers that cite POI
TotalPapersfromPOIasCol = data.iloc[:, POI_INDEX]
Counter(TotalPapersfromPOIasCol)
CitationsLevelAbove = [i for i, x in enumerate(TotalPapersfromPOIasCol) if x == 1]
#C
#print(CitationsLevelAbove)
print("CitationsLevelAbove- " , "C- " ,len(CitationsLevelAbove))

#%%

#ALl citations cited by papers that cite POI P_i
CitationsSameLevel = list()
for i in CitationsLevelAbove :
  # print(i)
  curr = data.iloc[i,:]
  # print(curr)
  cited_papers = [j for j, x in enumerate(curr) if x == 1 and j!=POI_INDEX]
  for y in cited_papers :
    CitationsSameLevel.append(y)

print("CitationsSameLevel-", " P_i - ", len(CitationsSameLevel))

#%%

#All citations of POI  Rf
TotalCitationsLevelBelow = data.iloc[POI_INDEX,:]
CitationsLevelBelow = [j for j, x in enumerate(TotalCitationsLevelBelow) if x == 1]
print("CitationsLevelBelow", " Rf ",len(CitationsLevelBelow))


#%%

#All papers that also cite the same papers that appear in the citations of POI P_j
#For each of the reference papers Rfi, extract all other papers Pj that cited Rf
CitationsLevelSame2 = list()
for i in CitationsLevelBelow :
  # print(i)
  curr = data.iloc[:,i]
  cited_i = [j for j, x in enumerate(curr) if x == 1 and j!=POI_INDEX]
  for y in cited_i :
    CitationsLevelSame2.append(y)

print("CitationsLevelSame2" , " P_j ", len(CitationsLevelSame2))
#print(CitationsLevelAbove2)

#%%

print("CitationsLevelSame" , " P_i ", len(CitationsSameLevel))
print("CitationsLevelSame" , " P_j ", len(CitationsLevelSame2))


#%%

#intersections of all arrays 
#making candicate papers
#Select all the candidate papers CP from Pi and Pj which are co-cited with the POI 
#and 
# which has been referenced by at least any of the POI references

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

CandidatePapers=intersection(CitationsSameLevel,CitationsLevelSame2)

#CP1 = intersection(CitationsLevelBelow,CitationsSameLevel)
#CP2 = intersection(CitationsLevelBelow,CitationsLevelAbove2)
#
#print(len(CP1),len(CP1))
#
#CP3 = list(set().union(CP1,CP2))
#
#print(len(CP3))
#
#CandidatePapers = [] 
#for i in CP3:
#  count = 0 
#  for j in CitationsLevelBelow:
#    if (data[j][i]==1):
#      CandidatePapers.append(i)
#      break
#      
#print("CandidatePapers- ", " CP - ", len(CandidatePapers))

#%%

#Similarity Matrix Co-Cited Matrix TABLE 1

CandidatePapers.insert(0,POI_INDEX)

print(CandidatePapers)
print(CitationsLevelAbove)


# ci sites cpj then 1 else 0
#Paper-citation relation matrixbased on co-occurred
CandidatePaperCitationMatrix = np.zeros((len(CandidatePapers),len(CitationsLevelAbove)))
print(CandidatePaperCitationMatrix.shape)

CandidatePaperCitationDataframe = pd.DataFrame(CandidatePaperCitationMatrix, index = CandidatePapers, columns= CitationsLevelAbove)
for i in CitationsLevelAbove:
  for j in CandidatePapers:
    if(data[j][i]==1):
      CandidatePaperCitationDataframe[i][j] = 1
print(CandidatePaperCitationDataframe)
#Due to convention: Row cites Column
CandidatePaperCitationDataframe=CandidatePaperCitationDataframe.T

#%%

#Similarity Matrix Co-Referenced MAtrix TABLE 3

#CandidatePapers.insert(0,POI_INDEX)

print(CandidatePapers)
print(CitationsLevelBelow)


# ci sites cpj then 1 else 0
#Paper-citation relation matrixbased on co-occurred
CandidatePaperReferenceMatrix = np.zeros((len(CandidatePapers),len(CitationsLevelBelow)))
print(CandidatePaperReferenceMatrix.shape)

CandidatePaperReferenceDataframe = pd.DataFrame(CandidatePaperReferenceMatrix, index = CandidatePapers, columns= CitationsLevelBelow)
for i in CitationsLevelBelow:
  for j in CandidatePapers:
    if(data[i][j]==1):
      CandidatePaperReferenceDataframe[i][j] = 1
print(CandidatePaperReferenceDataframe)

#%%

#Calculating Cosine Similarity Matrix
cosinMatrixCandidatePaperCitationDataframe = 1-pairwise_distances(CandidatePaperCitationDataframe.T, metric="cosine")
cosinMatrixCandidatePaperReferenceDataframe = 1-pairwise_distances(CandidatePaperReferenceDataframe, metric="cosine")



#%%

CitationsSelectedPapers = {}
arr = cosinMatrixCandidatePaperCitationDataframe[0]
for i in range(len(arr)):
    print(CandidatePapers[i], " " , arr[i])
    CitationsSelectedPapers[CandidatePapers[i]]= arr[i]
    

#%%

ReferenceSelectedPapers = {}
arr = cosinMatrixCandidatePaperReferenceDataframe[0]
for i in range(len(arr)):
    print(CandidatePapers[i], " " , arr[i])
    ReferenceSelectedPapers[CandidatePapers[i]]= arr[i]
    

#%%

FINALSELECTEDPAPERS = {}
for i in range(len(CitationsSelectedPapers)):
    #print(list(CitationsSelectedPapers.keys())[i])
    FINALSELECTEDPAPERS[list(CitationsSelectedPapers.keys())[i]] = (CitationsSelectedPapers[list(CitationsSelectedPapers.keys())[i]] + ReferenceSelectedPapers[list(ReferenceSelectedPapers.keys())[i]])/2
    
    
print(FINALSELECTEDPAPERS)
    
    
dict(sorted(FINALSELECTEDPAPERS.items(), key=lambda item: item[1], reverse = True))

#%%

print("Papers Recommended for Paper - ", POI_INDEX)
print(papers[POI_ID].title ," are- ")
topKPapers = 5
for i in range(1, topKPapers+1):
    pid = list(FINALSELECTEDPAPERS.keys())[i]
    for j in papers:
        if(papers[j].pid==pid):
            print(i, ". ", papers[j].title , " " , j)
#%%
