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


bmf=nimfa.Bmf(X0,rank=k,max_iter=12,lambda_w=1.1,lambda_h=1.1)

bmf_fit=bmf()

# W=bmf.W
# H=bmf.H
# print(W.shape())
# print(H.shape())
print(bmf_fit)
    # break



    #%%
matrix_bmf=bmf_fit.fitted()


# # np.save('matrix_factor.npy',X0)
if(np.all(matrix_bmf==0)):
    print('zero')
else:
    print("some 1")


#%%
U=bmf.W
V=bmf.H
print(U.shape)
print(V.shape)
# print(X0[0])
# mat=U.dot(V)
# print(mat)
# d1=pd.DataFrame(mat)
data=pd.DataFrame(matrix_bmf)
# print(d1)
#%%
print(data)
#%%
for POI_ID in ['P10-1142','W11-2165']:
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
            
# print(final)
#%%
    