# %%
import random
import numpy as np
import pickle
with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

# NxD
train_x = train_data[:,:23]
factors = 1/np.amax(train_x,0)
train_x = train_x*factors

# Nx1
train_y = train_data[:,23] # labels are either 0 or 1
test_x = test_data[:,:23]
test_y = test_data[:,23] # labels are either 0 or 1