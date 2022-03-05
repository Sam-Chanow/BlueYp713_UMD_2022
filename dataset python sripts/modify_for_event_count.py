import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df=pd.read_csv('updated_data.csv', low_memory=False)
print("Dataset Head:")
print(df.head())

#Setup map of event type to id

event_types = df["event_type"].unique()
event_types_dict = dict()

for i, event_type in enumerate(event_types):
    event_types_dict[event_type] = i
print("Unique event ids:")
print(event_types_dict)

#create dictionary with fips county code mapped to list with counts of each event type
fips_event_counts = dict()

for fips in df["fips_county_code"].unique():
    fips_event_counts[fips] = [0] * len(event_types_dict)
#fips_event_counts['nan'] = [0] * len(event_types_dict)

for i in range(len(df["event_type"])): #number of rows in dataset
    row = df.iloc[i,:]
    event_type = row[7] #event type
    #print(event_type)
    fips = row[31] #fips county code
    #print(fips)
    if (not math.isnan(fips)): # and (not math.isnan(event_type)):
        fips_event_counts[fips][event_types_dict[event_type]] += 1

print(fips_event_counts)

for fips in fips_event_counts:
    m = max(fips_event_counts[fips])
    max_index = fips_event_counts[fips].index(m)
    fips_event_counts[fips] = event_types[max_index]

print(fips_event_counts)

fips_keys = list(fips_event_counts.keys())

for i, fips in enumerate(fips_keys):
    if i != 72:
        fips = str(int(fips))
        if len(fips) < 5: fips = "0" + fips
        fips_keys[i] = fips



df2 = pd.DataFrame({'fips_county_code': fips_keys, 'event_type':fips_event_counts.values()})
df2.to_csv('max_event_data.csv',index=False)


#print(df['event_type'].value_counts())
