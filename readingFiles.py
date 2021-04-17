# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:16:40 2021

@author: Sezal
"""
class paper:
  def __init__(self, ID, author, title, venue, year):
    self.id = ID
    self.author = author
    self.title = title
    self.venue = venue
    self.year = year
    
# Using readlines()
filename = 'aan/release/2014/acl-metadata.txt'
with open(filename, "r", encoding="ANSI") as f:
    lines = f.readlines()
 
count = 0

for line in lines:
    count+=1
    string = line.split('=')[1].strip()
    p = paper()
    if(count==1):
        p.id = string
        print(string)
    
        
    