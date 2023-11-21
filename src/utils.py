import os
import datetime
import random
import concurrent.futures
import mlflow
import dill

from sklearn.metrics import precision_score, recall_score, f1_score


def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)
        
def load_object(file_path):
    with open(file_path, 'rb') as file_obj:
        return dill.load(file_obj)

def evaluate_model(true, predicted):
    precision = precision_score(true, predicted, average="weighted")
    recall = recall_score(true, predicted, average="weighted")
    F1_score = f1_score(true, predicted, average="weighted")
    return precision, recall, F1_score


def train(model, X_train, X_test, y_train, y_test, PARAMS, idx, experiment_id):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, F1_score = evaluate_model(y_test, y_pred)
    metrics = {"Precision" : precision, "Recall" : recall, "F1_score" : F1_score}

    print(f"***\tRun_{idx}/{len(PARAMS)}\t***")
    print(metrics)

    date_time = datetime.datetime.now()
    date = date_time.strftime("%x")
    time_ = date_time.strftime("%X")

    # Start MLflow
    RUN_NAME = f"run_{idx} | {date} | {time_}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_params(PARAMS[idx])

        # Track metrics
        mlflow.log_metrics(metrics)

        # Track model
        mlflow.sklearn.log_model(model, "Classification")

def Best_in_experiment(experiment_name):
    print(f"***   ****   ****    ****    {experiment_name}    ****    ****   ****   ***")
    RUN = {}
    run = mlflow.search_runs(experiment_names=[experiment_name]).sort_values(by='metrics.F1_score', ascending=False)
    RUN["Model"] = experiment_name
    RUN["run_id"] = run.iloc[0]["run_id"]
    RUN["experiment_id"] = run.iloc[0]["experiment_id"]
    RUN["model_uri"] = "mlruns/" + RUN["experiment_id"]+ "/" + RUN["run_id"]+"/artifacts/Classification"
    RUN["F1_score"] = run.iloc[0]["metrics.F1_score"]
    RUN["Precision"] = run.iloc[0]["metrics.Precision"]
    dill.dump(RUN, open(f"BEST_MODELS\Best_{experiment_name}.pkl", "wb"))