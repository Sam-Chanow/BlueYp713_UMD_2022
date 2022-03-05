#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:01:02 2022

@author: m231026


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
import random as rand
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack, find
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

df=pd.read_csv('data.csv', low_memory=False)
print(df.head())

import requests
import urllib

#I know this is bad programming blahhh
def fips_from_latlong(lat, long):
    params = urllib.parse.urlencode({'latitude': lat, 'longitude': long, 'format':'json'})
    url = 'https://geo.fcc.gov/api/census/block/find?' + params

    response = requests.get(url)
    data = response.json()
    return data['County']['FIPS']

fips = [fips_from_latlong(lat, long) for lat, long in tqdm(zip(df['latitude'], df['longitude']))]

df["fips_county_code"] = fips
#counts = dict()
#for fip in fips:
    #if fip in counts:
        #counts[fip] += 1
    #else:
      #  counts[fip] = 1
#

df.to_csv('updated_data.csv', index=False)


# In[ ]:


#fips = list(counts.keys())
#values = list(counts.values())

#We need to add a column to the dataset that is the FIPS county code

#fips_density = pd.DataFrame({'fips': fips, 'density': values})
#fips_density.to_pickle('fips_density_dataset.pkl')