# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:19:21 2021

@author: Raghav Rathi
"""

"""Binary Matrix Factorisation"""
import nimfa

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


bmf=nimfa.Bmf(X0,rank=20,max_iter=10,lambda_w=1.1,lambda_h=1.1)

fit=bmf.factorize()
# W=bmf.W
# H=bmf.H
# print(W.shape())
# print(H.shape())
print(fit)
    # break



#%%
matrix=fit.fitted()


# # np.save('matrix_factor.npy',X0)
if(np.all(matrix==0)):
    print('zero')
else:
    print("some 1")


#%%
U=bmf.W
V=bmf.H
print(U.shape)
print(V.shape)
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
print(final_dic)
#%%
print("Papers Recommended for Paper - ", POI_ID)
print(papers[POI_ID].title ,": \n"," are- ")
topKPapers = 5
for i in range(1, topKPapers+1):
    pid = list(final_dic.keys())[i]
    j=inverse_id[pid]
    print(i, ". ", papers[j].title , " " , j)
            
# print(final)
#%%
    