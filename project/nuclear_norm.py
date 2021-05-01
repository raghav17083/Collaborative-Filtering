# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:07:16 2021

@author: Raghav Rathi
"""

from paper_class import paper
import numpy as np
from tqdm import tqdm


from scipy.sparse import linalg as slinalg
import copy
import pandas as pd
import copy
#%%
papers={}
inverse_id={}
with open("datasets_inUse/paper_ids.txt","r", encoding="utf8") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # making the entire title sentence
        title=' '.join(l[1:len(l)-1])
        # paper id pid is increasing values of 1 with eveyr loop
        papers[l[0]]=paper(pid, l[0], title, l[-1])
        inverse_id[pid]=l[0]
        pid+=1
        
#%%

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

#%%

pre_matrix=paper_citation_matrix()


#%%
"""Nuclear Norm algorithm"""
thres=0.2
lamda=0.25
iters=5

X=pre_matrix
for i in tqdm(range(iters),leave=True,position=0):
    """B is initialised in each iter"""
    # B=Xfin+(Y-np.multiply(R,Xfin))
    B=X
    """U, S, V found using SVD. Dimensions of U=943x943, S=1,943, V=1682,1682"""
    u,s,v=slinalg.svds(B)
    
    """Soft Threshold, value=s-lambda/2 if s>lambda/2 else 0"""
    s = [x-lamda/2 if x-lamda/2 >0 else 0 for x in s]
    s=np.diag(s)
    
    """Creating a padded matrix with dimension 943x1682"""
    # sigma=np.zeros((X.shape[0],X.shape[1]))
    
    """Fill only first n (943) diagonal elements of sigma with elements of S"""
    # np.fill_diagonal(sigma,s)
    
    
    X=np.matmul(u,np.matmul(s,v))
        
    """Break if converged"""
    # 
#%%
matrix=copy.deepcopy(X)
data=pd.DataFrame(matrix)

#%%
for POI_ID in ['P10-1142','P12-1041']:
#POI_ID = "P12-1041"
    poi_index=papers[POI_ID].pid
    x=data.iloc[poi_index]
    print(x) # papers that POI cites
    print(len(x))

    final={}
    
    for i in range(len(x)):
        final[i]=x[i]
        
    final=dict(sorted(final.items(), key=lambda item: item[1],reverse=True))
    # print(final_dic)
    print("Papers Recommended for Paper - ", POI_ID)
    print(papers[POI_ID].title ,": \n"," are- ")
    topKPapers = 10
    for i in range(1, topKPapers+1):
        pid = list(final.keys())[i]
        j=inverse_id[pid]
        print(i, ". ", papers[j].title , " " , j)
