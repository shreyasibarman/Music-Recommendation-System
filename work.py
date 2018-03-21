import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
import time

user_data_filepath = '/home/shreyasi/Desktop/SVDshit/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv'
user_profiles_filepath = '/home/shreyasi/Desktop/SVDshit/lastfm-dataset-360K/usersha1-profile.tsv'

df_data = pd.read_table(user_data_filepath, header = None, 
                        names = ['user', 'musicbrainz-artist-id', 'artist', 'plays'],
                        usecols = ['user', 'artist', 'plays'])

df_user = pd.read_table(user_profiles_filepath,
                          header = None,
                          names = ['user', 'gender', 'age', 'country', 'signup'],
                          usecols = ['user', 'gender','country'])

len(df_data)
len(df_user)

df_user_IN = df_user[df_user['country']=='New Zealand'].drop('country',axis=1)
df_user_IN_female = df_user_IN[df_user_IN['gender']=='f'].drop('gender',axis=1)
print(len(df_user_IN_female))

df = df_data.merge(df_user_IN_female, left_on='user', right_on='user', how='inner')
df = df.groupby(['user','artist'], as_index=False).sum()
len(df)

df_artist = df_data.groupby(['artist'])['plays'].sum().reset_index().rename(columns = {'plays':'total_plays'})
df_artist.describe()

df_artist['total_plays'].quantile(0.99)

df_top_artist = df_artist[df_artist['total_plays']>200000].sort_values('total_plays', ascending=False)
print('Top 10 artists: \n', df_top_artist[0:9])

top_artist= list(df_top_artist['artist'])
df=df[df['artist'].isin(top_artist)]

df.head()
len(df)

matrix = df.pivot(index='artist',columns='user',values='plays').fillna(0)
matrix_sparse = csr_matrix(matrix)

matrix.shape
matrix.index.get_loc('radiohead')
#matrix.index[1976]
#matrix.iloc[1976]
matrix.loc['radiohead']

item_similarity = pairwise_distances(matrix_sparse,metric='cosine')
user_similarity = pairwise_distances(matrix_sparse.T,metric='cosine')

item_similarity.shape
user_similarity.shape

def predict(matrix, similarity, type='user'):
    if type=='user':
        mean_user_rating = matrix.mean(axis=1)
        ratings_diff = (matrix - mean_user_rating)
        pred = mean_user_rating + similarity.dot(ratings_diff)/ np.array([np.abs(similarity).sum(axis=1)]).T
    elif type== 'item':
        pred = matrix.dot(similarity)/ np.array([np.abs(similarity).sum(axis=1)])
    return pred


item_prediction = predict(matrix_sparse.T, item_similarity, type='item')
user_prediction = predict(matrix_sparse.T, user_similarity, type='user')


