import sys
sys.path.append(r'C:\Users\user_\OneDrive\Bureau\Mirakl\Use case (Mirakl)')

from src.experiences import AdaBoostClassifier_experience, DecisionTree_experience, KNeighborsClassifier_experience, RandomForestClassifier_experience, svc_experience, XGBClassifier_experience
import pandas as pd
from sklearn.model_selection import train_test_split

import time
import random
import concurrent.futures
import pandas as pd
import dill

import mlflow


from xgboost import XGBClassifier

from src.utils import Best_in_experiment

if __name__ == "__main__":
    train_data = pd.read_csv("data\Train_Reducer_data_FastICA.csv")
    X_train = train_data.drop("category_id_target", axis=1)
    y_train = train_data["category_id_target"]
    
    
    test_data = pd.read_csv("data\Test_Reducer_data_FastICA.csv")
    X_test = test_data.drop("category_id_target", axis=1)
    y_test = test_data["category_id_target"]    
    
    
    print("\n######################################    Experience 1   ######################################\n")
    KNeighborsClassifier_experience.KNeighborsClassifier_experience(X_train, X_test, y_train, y_test)
    print("\n######################################    Experience 2   ######################################\n")
    DecisionTree_experience.Decision_Tree_experience(X_train, X_test, y_train, y_test)
    print("\n######################################    Experience 3   ######################################\n")
    RandomForestClassifier_experience.RandomForestClassifier_experience(X_train, X_test, y_train, y_test)
    print("######################################    Experience 4   ######################################\n")
    AdaBoostClassifier_experience.AdaBoostClassifier_experience(X_train, X_test, y_train, y_test)
    print("\n######################################    Experience 5   ######################################\n")
    XGBClassifier_experience.XGBClassifier_experience(X_train, X_test, y_train, y_test)
    
    
    EXPERIMENTS = [ "KNeighborsClassifier - Experiment", "DecisionTreeRegressor - Experiment", "RandomForestClassifier - Experiment", "AdaBoostClassifier - Experiment","XGBoostClassifier - Experiment"]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = executor.map(Best_in_experiment, EXPERIMENTS)
        
        
    experiments_infos = pd.DataFrame(columns=['Model', 'run_id','experiment_id','model_uri','Precision','F1_score'])
    for i in EXPERIMENTS:
        dic_ = dill.load(open(f"BEST_MODELS\Best_{i}.pkl", "rb"))        
        dic_1 = pd.DataFrame([list(dic_.values())],columns = list(dic_.keys())) 
        experiments_infos = pd.concat([experiments_infos, dic_1], ignore_index=True)

    
    experiments_infos = experiments_infos.sort_values(by='F1_score', ascending=False)
    experiments_infos.to_csv('experiments_infos.csv', index=False)

    experiments_infos = pd.read_csv("experiments_infos.csv")
    
    print("\n\n\n\t\t\t**    **  **  Show experiments infos  **  **    **\n\n")
    print(experiments_infos.head())
    print("\n\n\n")
    
    model_uri = experiments_infos.iloc[0]["model_uri"]
