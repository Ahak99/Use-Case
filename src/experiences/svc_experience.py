import sys
sys.path.append(r'C:\Users\user_\OneDrive\Bureau\Mirakl\Use case (Mirakl)')
import time
import random
import concurrent.futures
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.utils import train



def SupportVectorClassifier_experience(X_train, X_test, y_train, y_test):
    EXPERIMENT_NAME = "SupportVectorClassifier - Experiment"
    # EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    EXPERIMENT_ID=current_experiment['experiment_id']

    # Elements of experience
    C = [random.uniform(9000, 9200) for _ in range(20)]
    epsilon = [random.uniform(0.01, 0.5) for _ in range(20)]
    kernel = ["linear", "rbf", "sigmoid"]

    parm_list = []
    for c in C:
        for k in kernel:
            parm_list.append([c, k])

    PARAMS = {}
    MODELS = {}
    for i in range(len(parm_list)):
        PARAMS[i+1] = {"C" : parm_list[i][0], "kernel" : parm_list[i][1]}
        MODELS[i+1] = {"model": SVC(C = parm_list[i][0], kernel = parm_list[i][1])}

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
    
    SupportVectorClassifier_experience(X_train, X_test, y_train, y_test)