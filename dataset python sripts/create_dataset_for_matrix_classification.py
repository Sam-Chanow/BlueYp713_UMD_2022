import numpy as np
import pandas as pd


df=pd.read_csv('updated_data.csv', low_memory=False)
print(df.head())

location = list(df["fips_county_code"])
month = list(df["event_date"])
assoc = list(df["assoc_actor_1"])
event_type = list(df["sub_event_type"])


#Get month numerically
month_dict = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
month = [month_dict[x.split("-")[1]] for x in month]

#Decrease value of Fips county codes so they dont overwhelm other data
location = [float(x) / 1000.0 for x in location]

#Get assoc actor numerically
actors = df["assoc_actor_1"].unique()
actor_dict = dict()

for i, actor in enumerate(actors):
    actor_dict[actor] = i + 1

assoc = [actor_dict[x] / 10 for x in assoc] #Added / 10 to keep it in the same order of magnitude as the other values

#Get event type numerically
events = df["sub_event_type"].unique()
event_dict = dict()

for i, event in enumerate(events):
    event_dict[event] = i + 1

event_type = [event_dict[x] for x in event_type]

df = pd.DataFrame({'location': location, 'month': month, 'actor':assoc, 'event_sub_type':event_type})

df.to_csv('matrix_dataset.csv', index=True)
