#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd


# In[3]:


play_list1 = pd.read_csv('df_playlist.csv',index_col=0)


# In[4]:


play_list2 = pd.read_csv('df_playlist1.csv',index_col=0)


# In[7]:


play_list = [play_list1,play_list2]


# In[8]:


playlist = pd.concat(play_list)


# In[10]:


playlist.to_csv('final-playlist.csv')


# In[20]:


playlist


# In[18]:


playlist.columns


# In[34]:


feature = playlist.drop(['song','artist','type','id','uri','track_href','analysis_url','duration_ms','time_signature'],axis=1 )


# In[35]:


feature


# In[36]:


feature.hist(xlabelsize=8, figsize=(12,10))


# In[14]:


from sklearn.cluster import KMeans


# In[49]:


from sklearn.preprocessing import StandardScaler


# In[50]:


scaler = StandardScaler()


# In[51]:


audio_feature = scaler.fit_transform(feature)


# In[52]:


pd.DataFrame(audio_feature)


# In[53]:


kmeans = KMeans()


# In[54]:


kmeans.fit(audio_feature)


# In[55]:


clusters = kmeans.predict(audio_feature)

len(clusters)


# In[45]:


pd.Series(clusters).value_counts()


# In[56]:


playlist["clusters"] = clusters


# In[69]:


playlist.to_csv("final_list.csv")


# In[65]:


import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# In[66]:


spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
client_id = "5b1d2c1994a5431e8005aa884c463f4a",
client_secret = "cc7d916fce5749e2a548b3f88f8eb793"))


# In[ ]:





# In[ ]:




