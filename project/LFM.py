# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:03:12 2021

@author: Raghav Rathi
"""

from paper_class import paper
import numpy as np
from tqdm import tqdm


from scipy.sparse import linalg as slinalg
import copy
import pandas as pd
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
k=20

X0=copy.deepcopy(pre_matrix)
iters=10
for i in tqdm(range(iters),leave=True,position=0):
    u,sig,vh=slinalg.svds(X0,k=k)
    # print(u.shape,s.shape,v.shape)
    U=u[:,:k]
    sig=sig[:k]
    vh=vh[:k]   
    sig=np.diag(sig)
    V=np.matmul(sig,vh)
    
    
    # print(U)
    # print(V)
    U[U<0]=0
    V[V<0]=0
    
    for j in range(10):
        U=np.matmul(X0,np.linalg.pinv(V))
        U[U<0]=0
        V=np.matmul(np.linalg.pinv(U),X0)
        V[V<0]=0
    X0=U.dot(V)
    

    # break
#%%
np.save('matrix_factor.npy',X0)
#%%

matrix=np.load('matrix_factor.npy')

# print(X0[0])
data=pd.DataFrame(matrix)
# print(data)
#%%

POI_ID = "P12-1041"
poi_index=papers[POI_ID].pid
x=data[poi_index]
print(x) # papers citing POI
print(len(x))
#%%
final={}

for i in range(len(x)):
    final[i]=x.iloc[i]
    
final_dic=dict(sorted(final.items(), key=lambda item: item[1],reverse=True))

#%%
print("Papers Recommended for Paper - ", POI_ID)
print(papers[POI_ID].title ,": \n"," are- ")
topKPapers = 5
for i in range(1, topKPapers+1):
    pid = list(final_dic.keys())[i]
    j=inverse_id[pid]
    print(i, ". ", papers[j].title , " " , j)
            
# print(final)
    