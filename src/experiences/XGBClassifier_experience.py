import sys
sys.path.append(r'C:\Users\user_\OneDrive\Bureau\Mirakl\Use case (Mirakl)')
import time
import random
import concurrent.futures
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils import train
 

def XGBClassifier_experience(X_train, X_test, y_train, y_test):
    EXPERIMENT_NAME = "XGBoostClassifier - Experiment"
    # EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    EXPERIMENT_ID=current_experiment['experiment_id']

    # Elements of experience
    learning_rate = [random.uniform(0.001, 0.1) for _ in range(7)]
    n_estimators = [random.randint(100,200) for _ in range(7)]
    max_depth = [5,6,7]


    parm_list = []
    for e in n_estimators:
        for lr in learning_rate:
            for d in max_depth:
                parm_list.append([e, lr, d])

    PARAMS = {}
    MODELS = {}
    for i in range(len(parm_list)):
        PARAMS[i+1] = {"n_estimators" : parm_list[i][0], "learning_rate" : parm_list[i][1] , "max_depth" : parm_list[i][2]}
        MODELS[i+1] = {"model": XGBClassifier(n_estimators = parm_list[i][0], learning_rate = parm_list[i][1] , max_depth = parm_list[i][2])}

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
    
    
# if __name__ == "__main__":
#     data = pd.read_csv("data\Reducer_data.csv")
#     X = data.drop("category_id_target", axis=1)
#     Y = data["category_id_target"]
#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X,list(Y),test_size=0.2,random_state=42)
#     # X_train_reducer, X_test_reducer, y_train, y_test = train_test_split(X,list(Y),test_size=0.2,random_state=42)
    
#     XGBClassifier_experience(X_train, X_test, y_train, y_test)