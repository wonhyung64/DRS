#%%
import os 
import numpy as np
import pandas as pd


# %%
path = "./data/KuaiRec/data"
small = pd.read_csv(f"{path}/small_matrix.csv")
big = pd.read_csv(f"{path}/big_matrix.csv")

small["interaction"] = small["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)
big["interaction"] = big["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)

test = small[["user_id", "video_id", "interaction"]].to_numpy()
train = big[["user_id", "video_id", "interaction"]].to_numpy()
