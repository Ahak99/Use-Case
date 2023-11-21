# Import librairies
import sys
sys.path.append(r'C:\Users\user_\OneDrive\Bureau\Mirakl\Use case (Mirakl)')
import time
import random
import concurrent.futures
import pandas as pd
import mlflow

from src.utils import save_object, load_object
from sklearn.decomposition import FastICA, KernelPCA

# Data path
train_path = "data\data_train_final.csv"
test_path = "data\data_test_final.csv"

print("Start reading data\n")
print("*"*50)
# Read Data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print("*"*50)

# Dataset preparation
train_data.drop("product_id", axis=1, inplace=True)
test_data.drop("product_id", axis=1, inplace=True)

# Trasnfor data (represent "category_id" by values between 0-100)
dic = {}
category_id_list = sorted(list(set(train_data["category_id"].values)))
for i in range(len(category_id_list)):
  dic[category_id_list[i]] = i

# trasformation for training data
train_data["category_id_target"] = train_data["category_id"]
train_data["category_id_target"].replace(dic, inplace=True)
# trasformation for testing data
test_data["category_id_target"] = test_data["category_id"]
test_data["category_id_target"].replace(dic, inplace=True)

# Divide data into explanatory variables and target variables
X_train = train_data.drop(["category_id", "category_id_target"], axis = 1)
Y_train = train_data["category_id_target"]

X_test = test_data.drop(["category_id", "category_id_target"], axis = 1)
Y_test = test_data["category_id_target"]


# Start Model building and training

# Dimensionality reduction
# 1. Use FastICA (Fast algorithm for Independent Component Analysis)
print("\nStart Dimensionality reduction\n")

# KernelPCA_reducer = KernelPCA(n_components=12, kernel='rbf')
# KernelPCA_reducer = KernelPCA_reducer.fit(X_train)
# print("*"*50)
# X_train_reduced = KernelPCA_reducer.transform(X_train)

# save_object(
#               file_path = ".\KernelPCA_reducer.pkl",
#               obj = KernelPCA_reducer
#             )
# ####################################################  FastICA  ##################################################################
FastICA_reducer = FastICA(n_components=12, random_state=0, whiten='unit-variance', whiten_solver = "eigh")
FastICA_reducer = FastICA_reducer.fit(X_train)
print("*"*50)
X_train_reduced = FastICA_reducer.transform(X_train)

save_object(
              file_path = ".\FastICA_reducer.pkl",
              obj = FastICA_reducer
            )
# ##################################################################################################################################

df_X_train_reduced = pd.DataFrame(X_train_reduced, columns=["Col_1", "Col_2", "Col_3", "Col_4", "Col_5", "Col_6", "Col_7", "Col_8", "Col_9", "Col_10", "Col_11", "Col_12"])
# Concatenate new dataframe, with reduced dimension (12 instead of 128)
Reduced_df = pd.concat([df_X_train_reduced, Y_train], axis=1)
Reduced_df.to_csv("data\Train_Reducer_data_FastICA.csv", index=False)

print("Show training data reduced\n")
print(Reduced_df.head)

X_test_reduced = FastICA_reducer.transform(X_test)

df_X_test_reduced = pd.DataFrame(X_test_reduced, columns=["Col_1", "Col_2", "Col_3", "Col_4", "Col_5", "Col_6", "Col_7", "Col_8", "Col_9", "Col_10", "Col_11", "Col_12"])
# Concatenate new dataframe, with reduced dimension (12 instead of 128)
Reduced_df = pd.concat([df_X_test_reduced, Y_test], axis=1)
Reduced_df.to_csv("data\Test_Reducer_data_FastICA.csv", index=False)

print("Show testing data reduced\n")
print(Reduced_df.head)