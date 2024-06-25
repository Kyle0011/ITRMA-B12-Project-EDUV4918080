#%%
import pandas as pd
import numpy as np
import sqlite3
# %%
df =pd.read_csv('heart.csv',sep=';')
# %%
df.columns =df.columns.str.strip()


# %%
#Create db/Connection

connection = sqlite3.connect('Hearts_DB.db')
# %%
# Load data to sqlite
df.to_sql(name='Heart_Disease',con=connection, if_exists='replace')
# %%
help(pd.read_csv)
# %%
connection.close()
