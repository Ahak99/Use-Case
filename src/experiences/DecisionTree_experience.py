import sys
sys.path.append(r'C:\Users\user_\OneDrive\Bureau\Mirakl\Use case (Mirakl)')
import time
import random
import concurrent.futures
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.utils import train


def Decision_Tree_experience(X_train, X_test, y_train, y_test):
    EXPERIMENT_NAME = "DecisionTreeRegressor - Experiment"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    # current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    # EXPERIMENT_ID=current_experiment['experiment_id']

    # Elements of experience
    splitter = ["best", "random"]
    max_depth = [3,4,5,6,7,8]
    min_samples_split = [1,2,3,4,5,6,7,8]

    parm_list = []
    for d in max_depth:
      for s in splitter:
        for ss in range(1,len(min_samples_split)):
          parm_list.append([s, d, ss])

    PARAMS = {}
    MODELS = {}
    for i in range(len(parm_list)):
        PARAMS[i+1] = {"splitter":parm_list[i][0], "max_depth":parm_list[i][1], "min_samples_split":parm_list[i][2]}
        MODELS[i+1] = {"model": DecisionTreeClassifier(random_state=0, splitter = parm_list[i][0], max_depth=parm_list[i][1], min_samples_split=parm_list[i][2])}

    print(f"***\t***\tSTART {EXPERIMENT_NAME}\t***\t***")
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pars = [(MODELS[i]['model'], X_train, X_test, y_train, y_test, PARAMS, i, EXPERIMENT_ID) for i in range(1, len(PARAMS)+1)]
        futures = executor.map(train, *zip(*pars))
    end = time.time()

    print(f"***\t***\tFINISH {EXPERIMENT_NAME}\t***\t***")

    run_time = end - start

    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)

    print(f"Run time execution : {hours}:{minutes}:{seconds}")



if __name__ == "__main__":
    data = pd.read_csv("data\Reducer_data.csv")
    X = data.drop("category_id_target", axis=1)
    Y = data["category_id_target"]
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,list(Y),test_size=0.2,random_state=42)
    # X_train_reducer, X_test_reducer, y_train, y_test = train_test_split(X,list(Y),test_size=0.2,random_state=42)
    
    Decision_Tree_experience(X_train, X_test, y_train, y_test)