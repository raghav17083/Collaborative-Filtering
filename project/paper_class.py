# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:05:04 2021

@author: Raghav Rathi
"""
class paper:
  """author : list of authors"""
  def __init__(self,pid, ID, title, year, author, venue):
    self.pid = pid
    self.ID = ID
    self.title = title
    self.year = year
    self.author = author
    self.venue = venue
