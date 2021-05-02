# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:35:45 2021

@author: Raghav Rathi
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:19:21 2021

@author: Raghav Rathi
"""

"""Sparse Nonnegative Matrix Factorization SNMF"""
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
k=15
X0=copy.deepcopy(pre_matrix)
def SNMF(X,k=20):


    snmf=nimfa.Snmf(X0,rank=k,max_iter=10)
    
    snmf_fit=snmf()
    
    # W=bmf.W
    # H=bmf.H
    # print(W.shape())
    # print(H.shape())
    # break
    target=snmf_fit.fitted()
    return target

    #%%

matrix_snmf=SNMF(X0,k)
# # np.save('matrix_factor.npy',X0)
if(np.all(matrix_snmf==0)):
    print('zero')
else:
    print("some 1")

#%%
"""MAE Calculation"""

"""Calculate MAE"""
cnt=0
sum_mae=0
for i in range(nop):
    for j in range(nop):
        if(pre_matrix[i][j]==1):
            cnt+=1
            sum_mae+=abs(matrix_snmf[i,j]-pre_matrix[i,j])
            # print(cnt)
        
print(sum_mae/cnt)

#%%
data=pd.DataFrame(matrix_snmf)
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

snmf_matrix=SNMF(matrix,20)

data_snmf=pd.DataFrame(snmf_matrix)
k_list=[]
for k in range(2,topK,2):
    recommended=[] ## D
    final={}
    x=data_snmf.iloc[poi_index]
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

    