#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:04:29 2022

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
import plotly.figure_factory as ff

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

#fips_density = pd.read_pickle("fips_density_dataset.pkl")
import pickle5 as pickle
#fips_density = None
with open("fips_density_dataset.pkl", "rb") as fh:
  fips_density = pickle.load(fh)
print(fips_density.head())
fips = list(fips_density["fips"])
values = list(fips_density["density"])

fips_density['density'] = fips_density['density'].astype('int64', copy=False)
print(fips_density['fips'][0])

import plotly.express as px
fig = px.choropleth(fips_density, geojson=counties, locations='fips', color='density',
                           color_continuous_scale="turbo",#"Viridis",
                           range_color=(1, 100),
                           scope="usa",
                           labels={'density':'# of events'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()