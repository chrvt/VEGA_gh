# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:39:17 2021

@author: horvat
"""

#import .csv data

import pandas as pd
line_break = 70  #after how many chars should be a line break?
break_symbol = "$"

user_comments = pd.read_csv("D:\PROJECTS\VEGA_gh\projects\CC_user_comments\comments.csv")

comments = user_comments["Comment"]

for k_comm in range(comments.shape[0]): #run through all comments
    comm = comments[k_comm]
    for k in range(len(comm)):  #go through alls chars in comment
        if k % line_break == 0 and k>0:
            n_char = k+comm[k:].find(" ")
            # line[:10].replace(';', ':') + line[10:] comm[n_char:n_char+2].replace(" ",'4')
            comm = comm[:n_char] + break_symbol  + comm[n_char+1:]
    comments[k_comm] = comm
           

user_comments["Comment"] = comments

user_comments.to_csv('comments_linebreak.csv')