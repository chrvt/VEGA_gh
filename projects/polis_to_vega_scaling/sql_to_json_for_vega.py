#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:43:31 2022
    - takes .json table as input
    - returning PCA coordinates for plotting
    - creating new .json file for vega to process
@author: chrvt
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# df = pd.read_csv('maas_user_polls.csv',index_col='user')
# df_comments = pd.read_csv('maas_polls.csv',index_col='poll_id')
df = pd.read_csv('participants-votes.csv',index_col='participant')
df_comments = pd.read_csv('comments.csv',index_col='comment-id')

df_comments.index = df_comments.index.astype(str)

#filter out potential columns from ana

metadata_fields = ['group-id', 'n-comments', 'n-votes', 
                    'n-agree', 'n-disagree']
val_fields = [c for c in df.columns.values if c not in metadata_fields]


# remove statements (columns) which were moderated out
statements_all_in = sorted(list(df_comments.loc[df_comments["moderated"] > 0].index.array), key = int)


#val_fields = df.columns.values

#####SELECTING USERS#####
## for a row, count the number of finite values
def count_finite(row):
    finite = np.isfinite(row[val_fields]) # boolean array of whether each entry is finite
    return sum(finite) # count number of True values in `finite`

## REMOVE PARTICIPANTS WITH LESS THAN N VOTES check for each row if the number of finite values >= cutoff
def select_rows(df, threshold=10):
    
    number_of_votes = df.apply(count_finite, axis=1)
    valid = number_of_votes >= threshold
    
    return df[valid]
    
df = select_rows(df)

#############################
#############################

#metadata = df[metadata_fields]
vals = df[val_fields]
## If the participant didn't see the statement, it's a null value, here we fill in the nulls with zeros
vals = vals.fillna(0)  #<---in paper: column mean, we don't need since there should be no 0 value
vals = vals.sort_values("participant")
vals_all_in = vals

# remove statements (columns) which were moderated out
statements_all_in = sorted(list(df_comments.index.array))


sorted(list(df_comments.loc[df_comments["moderated"] > 0].index.array), key = int)
vals_all_in = vals[statements_all_in]

###some statistics

melted = vals.melt();
all_votes = melted.count();
by_type = melted["value"].value_counts();
total_possible_votes = all_votes["value"];
total_agrees = by_type[1];
total_disagrees = by_type[-1];
total_without_vote = 0 # by_type[0];

print("Dimensions of matrix:", df.shape)
print("Dimensions of matrix:", vals.shape)
print("Total number of possible votes:", total_possible_votes)
print("Total number of agrees:", total_agrees)
print("Total number of disagrees:", total_disagrees)
print("Total without vote:", total_without_vote)
print("Percent sparse: ", total_without_vote / total_possible_votes,"%")


#replace polls by comments for pol.is version
def polis_pca(dataframe, polls=None, components=2):
    pca_object = PCA(n_components=components) ## pca is apparently different, it wants 
    pca_object = pca_object.fit(dataframe) ## .T transposes the matrix (flips it)
    coords = pca_object.transform(dataframe)
    
    #coords = pca_object.components_ ## isolate the coordinates and flip 
    explained_variance = pca_object.explained_variance_ratio_
    
    if polls is not None:
        polls_coords = pca_object.transform(polls)
    else: polls_coords = None
    
    return coords, explained_variance, polls_coords


# def c(comment, coords):
#     fig, ax = plt.subplots(figsize=(7,5))
#     plt.sca(ax)
#     colorMap = {-1:'#A50026', 1:'#313695', 0:'#FEFEC050'}
#     ax.scatter(
#         x=coords[:,0],
#         y=coords[:,1],
#         c=vals[str(comment)].apply(lambda x: colorMap[x]),
#         s=10
#     )
#     ax.set_title("\n".join(wrap(str(comment) + "  " + str(df_comments['comment-body'][comment]))), fontsize=14)
#     print('Colorcode is based on how voted not clustering!')
#     plt.show()
    
# add dummy users
#for k in range(df_comments.shape[0]):
    
df_polls = np.eye(vals_all_in.shape[1])
    
coords, variance, polls_coords  = polis_pca(vals_all_in, polls=df_polls)

plt.figure(figsize=(7, 5), dpi=80)
plt.scatter(
    x=coords[:,0], 
    y=coords[:,1], 
    #c=metadata['n-votes'],
    cmap="magma_r",
    s=5
)

plt.scatter(
    x=polls_coords[:,0], 
    y=polls_coords[:,1], 
    c='red',
    s=10
)
plt.colorbar()

##add pca coordinates to vega-dataframe
df_vega = vals_all_in #df
df_vega = df_vega.assign(x=coords[:,0])
df_vega = df_vega.assign(y=coords[:,1])


#############################
##########>>>CLUSTERING<<<

#given a dataframe, returns the found lables/clusters and the corresponding centers
def polis_kmeans_(dataframe, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataframe)
    
    return kmeans.labels_, kmeans.cluster_centers_

def plot_embedding_with_clusters(embedding_,labels_):
    print("Plotting PCA embeddings with K-means, K="+str(np.max(labels_)+1))
    fig, ax = plt.subplots(figsize=(7,5))
    plt.sca(ax)
    ax.scatter(
        x=embedding_[:,0],
        y=embedding_[:,1],
        c=labels_,
        cmap="tab20",
        s=5
    )
    #ax.set_title("", fontsize=14)
    

# Step 1 - get PCA clusters 
embedding, explained_variance , components= polis_pca(vals_all_in)  
print("Explained variance:", explained_variance)

# Step 2 - K-means with K=100     !!!only for many participants!!!
clusters_100, centers = polis_kmeans_(embedding,n_clusters=6)

#plot_embedding_with_clusters(embedding, clusters_100)

# Step 3 - find optimal K
from sklearn.metrics import silhouette_score 
silhoutte_star = -np.inf
for K in range(2,4):
    clusters_K, _ = polis_kmeans_(centers,n_clusters=K)
    silhouette_K = silhouette_score(centers, clusters_K)
    if silhouette_K >= silhoutte_star:
        K_star = K
        silhoutte_star = silhouette_K
        clusters_K_star = clusters_K
print('Optimal K-mean clusters for K=',str(K_star))
plot_embedding_with_clusters(centers,clusters_K_star)
 
# Step 4 - assign each voter to "optimal" cluster
clusters_star = np.zeros(len(clusters_100))
colors = ['green','steelblue','orange','']
for k in range(6):#
    #find all indices with clusters k and assign them new star label
    clusters_star[np.where(clusters_100==k)]  = clusters_K_star[k]

df_vega = df_vega.assign(cluster=clusters_star)



############################
######comment relevance#####

#Step 1: Calculate N_v(g,c), N(g,c), and P_v(g,c)
N_groups, N_comments = clusters_K_star.max()+1, len(statements_all_in)
N_v_g_c = np.zeros([3,N_groups,N_comments]) #create N matrix 
P_v_g_c = np.zeros([3,N_groups,N_comments])
N_g_c = np.zeros([N_groups,N_comments])
v_values = [-1,0,1]

for g in range(N_groups): 
    idx_g = np.where(clusters_star == g)[0] #get indices of cluster g; caution_ idx != participant id
    for c in range(N_comments):
        comment = statements_all_in[c] #comment id
        # import pdb
        # pdb.set_trace()

        df_c = vals_all_in[str(comment)].iloc[idx_g] #data frame: [participants of group g,comment c], 
        for v in range(3):
            v_value = v_values[v]
            N_v_g_c[v,g,c] = (df_c == v_value).sum() #counts all v_value votes in data frame df_c
        N_g_c[g,c] = N_v_g_c[0,g,c] + N_v_g_c[2,g,c] #total votes corresponds to votes with +1 or -1  
        
        for v in range(3):
            P_v_g_c[v,g,c] = (1 + N_v_g_c[v,g,c]) / (2 + N_g_c[g,c])
        
#Step2: calculate R_v(g,c)                           
R_v_g_c = np.zeros([3,N_groups,N_comments])
for g in range(N_groups): 
    for c in range(N_comments):
        for v in range(3):
           R_v_g_c[v,g,c] = P_v_g_c[v,g,c] / np.delete(P_v_g_c[v,:,c],g,0).sum()  # np.delete neglects all entries with group g                                           
                                                   

import scipy.stats as stats
from scipy.stats import hypergeom

v_values = [-1,0,1]
p_values = np.zeros([N_groups,N_comments,3])
for g in range(N_groups): 
    idx_g = np.where(clusters_star == g)[0] #get indices of cluster g; caution_ idx != participant id
    idx_g_not = np.where(clusters_star != g)[0] #get indices of rest
    for c in range(N_comments):
        comment = statements_all_in[c] #comment id
        
        for v in range(3):
            v_value = v_values[v]
            N_v = (vals_all_in[str(comment)] == v_value).sum()    #totol number of v votes in comment c
            N_rest = (vals_all_in[str(comment)]).count() - N_v    #total number of votes =  number of participants
            
            df_c = vals_all_in[str(comment)].iloc[idx_g]  #get data frame of group g for comment c
            N_v_in_g = (df_c == v_value).sum()
            N_g = (df_c).count()   
            
            [M, n, N] = [N_rest+N_v, N_v, N_g]  #hypergeometric distribution parameters https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
            x = range(N_v_in_g-1 ,N_g+1)
            prb = hypergeom.pmf(x, M, n, N).sum() #calculates P(X>=N_v_in_g), i.e. p-value. 
            p_values[g,c,v] = prb * R_v_g_c[v,g,c]
            
df_comments_vega = df_comments

df_comments_vega  = df_comments_vega .assign(x=polls_coords[:,0])
df_comments_vega  = df_comments_vega .assign(y=polls_coords[:,1])

           
## Lets print the most significant comments for each group!
for g in range(N_groups): 
    idx_ = np.argsort(p_values[g,:,0]) #take only the significant comments
    
    
    df_comments_vega = df_comments_vega.assign(**{f'group_{g}' : p_values[g,:,0] })
      
    
    #subframe = df_comments_vegafor .query("`{0}` == 'Large'".format(myvar))
    
    df_comments_vega.rename(columns={'name': 'group'+str('_')+str(g) } )
    
    print('----------------------------------')
    print('most significant comment for group ',g)
    print('----------------------------------')
    for i in range(1):
        print(' ')
        print(df_comments['comment-body'][idx_[i]])
        print(' ')
        
        #How to embed comments into plot?
        #idx_[i]
        #participant_comment = np.zeros([1,vals_all_in.shape[0]])
        #participant_comment[0,idx_[i]] = 1
        #latent_comment = pca_object.transform(participant_comment)
   
    
   
df_comments_vega.to_csv('comments_vega.csv')
df_vega.to_csv('user_data_vega.csv')
