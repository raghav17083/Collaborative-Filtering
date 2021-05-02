# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:19:21 2021

@author: Raghav Rathi
"""

"""Binary Matrix Factorisation"""
import nimfa

import numpy as np
from tqdm import tqdm


from scipy.sparse import linalg as slinalg
import copy
import pandas as pd
#%%
class paper:
  def __init__(self,pid, ID, title, year):
    self.pid = pid
    self.ID = ID
    self.title = title
    self.year = year
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



def BMF(X,k):
    bmf=nimfa.Bmf(X,rank=k,max_iter=12,lambda_w=1.1,lambda_h=1.1)
    
    bmf_fit=bmf()
    
    # W=bmf.W
    # H=bmf.H
    # print(W.shape())
    # print(H.shape())
    print(bmf_fit)
    # break

    matrix_bmf=bmf_fit.fitted()
    return matrix_bmf

#%%
X0=copy.deepcopy(pre_matrix)
matrix_bmf=BMF(X0,k)

# # np.save('matrix_factor.npy',X0)
if(np.all(matrix_bmf==0)):
    print('zero')
else:
    print("some 1")


#%%
data=pd.DataFrame(matrix_bmf)
# print(d1)

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
import matplotlib.pyplot as plt
import random
"""Calculate Precision/ Recall"""
topK=100
POI_ID='P10-1142'
matrix=copy.deepcopy(pre_matrix)
recall=[]
precision=[] 
poi_index=papers[POI_ID].pid
allones=[]
for i in range(nop):
    if(matrix[poi_index,i]==1):
        allones.append(i)
# print(testSet)

#%%
testSet=random.sample(allones,(len(allones)//5))
testId=[inverse_id[i] for i in testSet]

for i in range(len(testSet)):
    matrix[poi_index,testSet[i]]==0

bmf_matrix=BMF(matrix,20)

data_nnmf=pd.DataFrame(bmf_matrix)
k_list=[]
for k in range(2,topK,2):
    recommended=[] ## D
    final={}
    x=data_nnmf.iloc[poi_index]
    for i in range(len(x)):
        final[i]=x[i]
        
    final_dic=dict(sorted(final.items(), key=lambda item: item[1],reverse=True))
    topKPapers = k
    for i in range(1, topKPapers+1):
        pid = list(final_dic.keys())[i]
        j=inverse_id[pid]
        recommended.append(j)
    
    k_list.append(k)
    """Recall"""
    intersection=list(set(testId) & set(recommended)) # D inter T
    recall.append(len(intersection)/len(testId))
    
    """Precision"""
    precision.append(len(intersection)/k)    

    
plt.plot(k_list, recall, c='red', label='Recall')
plt.plot(k_list, precision, c='blue', label ='Precision')
plt.title('Recall and Precision Graph')
plt.xlabel('List of top K Recommended Papers')
    
plt.legend()
plt.show()       