# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:24:08 2021

@author: Raghav Rathi
"""
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 00:17:08 2021

@author: Sezal
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from paper_class import paper
import copy
import nimfa
paper_id={}
id_paper={}
inverse_id={}

with open("datasets_inUse/paper_ids.txt","r", encoding="utf8") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # making the entire title sentence
        title=' '.join(l[1:len(l)-1])
        # paper id pid is increasing values of 1 with eveyr loop
        obj=paper(pid, l[0], title, l[-1],[],"")
        paper_id[l[0]]=obj
        id_paper[pid]=obj
        inverse_id[pid]=l[0]
        pid+=1
 #%%
with open("datasets_inUse/acl-metadata.txt","r", encoding="utf8") as file:
    pid=0
    paper=""
    for i in file.readlines():
        if(i!='\n'):
            l=i.split('=')
            if(l[0]=="id "):
                l[1]=l[1].strip()
                paper=l[1][1:-1]
            if(l[0]=="author "):
                l[1] = l[1].strip()
                auth = l[1][1:-1]
                #print(auth)
                auth_list=[i.strip() for i in auth.split(';')]
                paper_id[paper].author=auth_list
            if(l[0]=="venue "):
                l[1] = l[1].strip()
                venue = l[1][1:-1]
                #print(auth)
                paper_id[paper].venue=venue
            if(l[0]=="year "):
                continue
                   
#%%
"""Weight based on author-citation"""

nop=len(paper_id)

""""Paper Citation matrix"""

def paper_citation_matrix():
    with open("datasets_inUse/paper-citation-network-nonself.txt",'r') as file:
        matrix=np.zeros((nop,nop))
        for i in tqdm(file.readlines()):
            l=i.split()
            #print(papers[l[0]].pid," " , papers[l[2]].pid," -------------------")
            
            matrix[paper_id[l[0]].pid,paper_id[l[2]].pid]=1
    return matrix


#%%
# pre_matrix=paper_citation_matrix()

# ct=0
# for i in tqdm(range(nop)):
#     for j in range(nop):
#         if(i!=j):
#             auth1=id_paper[i].author
#             auth2=id_paper[j].author
#             inter=list(set(auth1) & set(auth2))
            
#             if(len(inter)!=0):
#                 ct+=1
#                 pre_matrix[i][j]+=0.5

# #%%            
# np.save("weight_matrix.npy",pre_matrix)

#%%

weighted=np.load("weight_matrix.npy")
non_weighted=paper_citation_matrix()
# print(np.count_nonzero(weighted==1.5))

#%%

# print(np.where(pre_matrix[paper_id['P10-1142'].pid]==1.5))


# print(np.where(pre_matrix==0.5))
#%%

def nnmf(X0,k):
    nnmf=nimfa.Nmf(X0,rank=k,max_iter=10, lambda_w=0.8,lambda_h=0.8)
    fit_nnmf=nnmf()
    # print(fit_nnmf)
    matrix_nn=fit_nnmf.fitted()
    # break
    return matrix_nn
#%%
import matplotlib.pyplot as plt
import random
"""Calculate Precision/ Recall"""
def evaluation_nnmf(matrix,topK=100,POI_ID='P10-1142'):
    # matrix=copy.deepcopy(pre_matrix)
    recall=[]
    precision=[] 
    poi_index=paper_id[POI_ID].pid
    allones=[]
    for i in range(nop):
        if(matrix[poi_index,i]==1):
            allones.append(i)
    # print(testSet)
    testSet=random.sample(allones,(len(allones)//3))
    testId=[inverse_id[i] for i in testSet]
    
    for i in range(len(testSet)):
        matrix[poi_index,testSet[i]]==0
    
    nnmf_matrix=nnmf(matrix,12)
    
    data_nnmf=pd.DataFrame(nnmf_matrix)
    k_list=[]
    for k in range(2,topK,2):
        recommended=[] ## D
        final={}
        x=data_nnmf.iloc[poi_index]
        # print(len(x))
        for i in range(len(x)):
            final[i]=x[i]
        # print(final) 
        # print('\n')
        final_dic=dict(sorted(final.items(), key=lambda item: item[1],reverse=True))
        # print(final_dic)
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
        
    return k_list,recall,precision,intersection

#%%        


#%%
k_list_nw,recall_nw,precision_nw,inter_nw=evaluation_nnmf(non_weighted,POI_ID='P12-1041')

k_list_w,recall_w,precision_w,inter_w=evaluation_nnmf(weighted,POI_ID='P12-1041')

# plt.subplot(1,1,1)

plt.plot(k_list_nw, recall_nw, c='red',label='non-weighted')
plt.plot(k_list_w, recall_w, c='blue', label ='weighted')
plt.title('Recall Graphs')
plt.xlabel('List of top K Recommended Papers')
plt.legend()
plt.show() 


# ax[1].plot(k_list, precision_nw, c='red',label='non-weighted')
# ax[1].plot(k_list, precision_w, c='blue', label ='weighted')
# ax[1].set_title('precision Graphs')

    # ax[1].set(xlabel='List of top K Recommended Papers')
    
# ax[0].legend()
# ax[0].show() 

# ax[1].legend()
# ax[1].show()
# plt.subplot(1,1,1)
plt.plot(k_list_nw, precision_nw, c='red',label='non-weighted')
plt.plot(k_list_w, precision_w, c='blue', label ='weighted')
plt.title('Precision Graphs')
plt.xlabel('List of top K Recommended Papers')
plt.legend()
plt.show() 

 #%%
print("recall_nw", recall_nw)
print("recall_w", recall_w)
print("precision_nw", precision_nw)
print("precision_w", precision_w)


