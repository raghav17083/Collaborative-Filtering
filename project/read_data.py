# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:37:02 2021

@author: Raghav Rathi
"""

import numpy as np

global paper_id_dic={}
global paper_title_dic={}

with open("datasets_inUse/paper_ids.txt","r") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # print(l)
        title=' '.join(l[1:len(l)-1])
        # print(title)
        paper_id_dic[l[0]]=pid
        pid+=1
        
        
# print(paper_id_dic)

nop=len(paper_id_dic)


    
#%%

""""Paper Citation matrix"""
# import platform
# print(platform.architecture())

def paper_citation_matrix(path):
    with open(path,'r') as file:
        matrix=np.zeros((nop,nop))
        for i in file.readlines():
            l=i.split()
            matrix[paper_id_dic[l[0]],paper_id_dic[l[2]]]=1
# print(matrix)

"""Checking sparsity. Takes very long"""
cnt=0
for i in range(nop):
    for j in range(nop):
        if(matrix[i,j]!=0):
            cnt+=1
print(cnt)