# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:37:02 2021

@author: Raghav Rathi
"""
import pandas as pd

global paper_id_dic
paper_id_dic={}
global paper_title_dic
paper_title_dic={}

global id_paper_dic
id_paper_dic={}

with open("datasets_inUse/paper_ids.txt","r") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # print(l)
        title=' '.join(l[1:len(l)-1])
        # print(title)
        paper_id_dic[l[0]]=pid
        id_paper_dic[pid]=l[0]
        pid+=1
        
        
# print(paper_id_dic)

nop=len(paper_id_dic)
# print(paper_id_dic["P10-1142"])

    
#%%
import numpy as np
import pickle
from tqdm import tqdm

""""Paper Citation matrix"""
# import platform
# print(platform.architecture())

def paper_citation_matrix(path):
    with open(path,'r') as file:
        matrix=np.zeros((nop,nop))
        for i in tqdm(file.readlines()):
            l=i.split()
            matrix[paper_id_dic[l[0]],paper_id_dic[l[2]]]=1
    return matrix

"""Checking sparsity. Takes very long"""
matrix=paper_citation_matrix("datasets_inUse/paper-citation-network-nonself.txt")
# np.save("matrix", matrix)
print(matrix.shape)
data=pd.DataFrame(matrix)
print(data[1])

#%%
#user based
# print(matrix[paper_id_dic["C08-1069"],paper_id_dic["C04-1041"]])

        
        
    # print(rows)
    
    