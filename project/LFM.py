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
import nimfa

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

def nnmf(X0,k):
    nnmf=nimfa.Nmf(X0,rank=k,max_iter=10,lambda_w=0.8,lambda_h=0.8)
    fit_nnmf=nnmf()
    
    
    # print(fit_nnmf)
    matrix_nn=fit_nnmf.fitted()
    # break
    return matrix_nn
#%%


# # np.save('matrix_factor.npy',X0)
# if(np.all(matrix_nn==0)):
#     print('zero')
# else:
#     print("some 1")

#%%

# matrix=np.load('matrix_factor.npy')

# print(X0[0])
matrix_nn=nnmf(X0,20)
data_nn=pd.DataFrame(matrix_nn)
print(data_nn)
np.save('nnmf_matrix.npy',matrix_nn)


#%%
matrix_nn=np.load('nnmf_matrix.npy')
"""Calculate MAE"""
cnt=0
sum_mae=0
for i in range(nop):
    for j in range(nop):
        if(pre_matrix[i][j]==1):
            cnt+=1
            sum_mae+=abs(matrix_nn[i,j]-pre_matrix[i,j])
            # print(cnt)
        
print(sum_mae/cnt)

#%%

for POI_ID in ['P12-1041','P10-1142']:
    poi_index=papers[POI_ID].pid
    x=data_nn.iloc[poi_index]
    # print(x) # papers citing POI
    # print(len(x))

    final={}
    
    for i in range(len(x)):
        final[i]=x[i]
        
    final_dic=dict(sorted(final.items(), key=lambda item: item[1],reverse=True))

    print("Papers Recommended for Paper - ", POI_ID)
    print(papers[POI_ID].title ,": \n"," are- ")
    topKPapers = 15
    for i in range(1, topKPapers+1):
        pid = list(final_dic.keys())[i]
        j=inverse_id[pid]
        print(i, ". ", papers[j].title , " " , j)
    print("\n")
                
# print(final)

    
    
#%%
import matplotlib.pyplot as plt
import random
"""Calculate Precision/ Recall"""
def evaluation(pre_matrix,topK=100,POI_ID='P10-1142'):
    matrix=copy.deepcopy(pre_matrix)
    recall=[]
    precision=[] 
    poi_index=papers[POI_ID].pid
    allones=[]
    for i in range(nop):
        if(matrix[poi_index,i]==1):
            allones.append(i)
    # print(testSet)
    testSet=random.sample(allones,(len(allones)//5))
    testId=[inverse_id[i] for i in testSet]
    
    for i in range(len(testSet)):
        matrix[poi_index,testSet[i]]==0
    
    nnmf_matrix=nnmf(matrix,20)
    
    data_nnmf=pd.DataFrame(nnmf_matrix)
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
        
    return k_list,recall,precision
        
    plt.plot(k_list, recall, c='red', label='Recall')
    plt.plot(k_list, precision, c='blue', label ='Precision')
    plt.title('Recall and Precision Graph')
    plt.xlabel('List of top K Recommended Papers')
        
    plt.legend()
    plt.show()       
    


    