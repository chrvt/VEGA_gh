# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:58:48 2021
User and friends generating .csv file for vega viz
@author: horvat
"""

import numpy as np
import pandas
import os

#Hyperparameters
n_user = 51
max_friends = 20
max_visits = 1000
user_idx = np.arange(n_user)
save_path = r'D:\PROJECTS\VEGA_gh\projects\CC_user_map\data'

user = []
friends = []
visits = []
for k in range(n_user):
    #Step1: generate a number of friends
    n_friends = np.random.choice(np.arange(max_friends))
    #Sample friends for user k
    k_friends = np.random.choice(user_idx, size=n_friends)
    #Sample visits on profile
    k_visits = np.random.choice(np.arange(max_visits), size=n_friends)
    #update list for data table
    user += list(np.zeros(n_friends)+k)
    friends += list(k_friends)
    visits += list(k_visits)
    
df = pandas.DataFrame(data={"user": user, "friends": friends, "visits": visits})

df.to_csv(os.path.join(save_path,'user_friends.csv'), sep=',',index=False)
