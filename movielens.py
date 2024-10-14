#%%
import numpy as np
import pandas as pd


#%%
file_dir = '/Users/wonhyung64/Github/DRS/data/ml-1m/ratings.dat'
df = pd.read_csv(file_dir, delimiter="::", header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]
df = df.iloc[:, :3]

num_user = df["user_id"].max()
num_item = df["item_id"].max()

np.random.seed(0)
x_train, x_test = [], []
for u in range(1, num_user+1):
    u_i_indices = df[df["user_id"] == u].index.tolist()
    u_i_test_indices = np.random.choice(u_i_indices, int(np.round(len(u_i_indices) * 0.2)))
    u_i_train_indices = [idx for idx in u_i_indices if idx not in u_i_test_indices]
    x_train.append(df.to_numpy()[u_i_train_indices, :])
    x_test.append(df.to_numpy()[u_i_test_indices, :])

x_train = np.concatenate(x_train)
x_test = np.concatenate(x_test)

np.save("/Users/wonhyung64/Github/DRS/data/ml-1m/train.npy", x_train, allow_pickle=True)
np.save("/Users/wonhyung64/Github/DRS/data/ml-1m/test.npy", x_test, allow_pickle=True)
