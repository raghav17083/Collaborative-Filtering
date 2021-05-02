# -*- coding: utf-8 -*-
"""
Created on Sun May  2 00:17:08 2021

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
from paper_class import paper

    
papers={}

with open("datasets_inUse/paper_ids.txt","r", encoding="utf8") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # making the entire title sentence
        title=' '.join(l[1:len(l)-1])
        # paper id pid is increasing values of 1 with eveyr loop
        papers[l[0]]=paper(pid, l[0], title, l[-1],"","")
        pid+=1
        
with open("datasets_inUse/acl-metadata.txt","r", encoding="utf8") as file:
    pid=0
    flag = False
    ID = 0
    for i in file.readlines():
        #print(i)
        l=i.split('=')
        if(flag):
            if(l[0]=="author "):
                l[1] = l[1].strip()
                auth = l[1][1:-1]
                #print(auth)
                papers[ID].author=auth
            if(l[0]=="venue "):
                l[1] = l[1].strip()
                auth = l[1][1:-1]
                #print(auth)
                papers[ID].venue=auth
            if(l[0]=="year "):
                flag = False
                
        if(l[0]=="id "):
            l[1] = l[1].strip()
            ID = l[1][1:-1]
            #print(ID)
            flag=True
        
#%%


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

author_collaborations.txt
