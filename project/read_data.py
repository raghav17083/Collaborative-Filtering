# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:37:02 2021

@author: Raghav Rathi
"""

paper_id_dic={}
paper_title_dic={}

with open("datasets_inUse/paper_ids.txt","r") as file:
    pid=0
    for i in file.readlines():
        l=i.split()
        # print(l)
        title=' '.join(l[1:len(l)-1])
        # print(title)
        paper_id_dic[l[0]]=pid
        pid+=1
        
        
        
print(paper_id_dic)
    
    