from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
#ClientAPI: lets us manage registered models and their aliases
from mlflow import MlflowClient

#Configure MLFlow tracking
mlflow.set_tracking_uri("http://127.0.0.1:7001/")

# Group related training runs under one experiment
mlflow.set_experiment("iris-classification")

#load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=42)

n_estimators = 10
random_state = 42

model_parameters={
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.65
}

with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision}")
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Recall: {recall}")

    #Model quality gate
    #Define the minimum acceptable accuracy for now
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(f"Model rejected: test accuracy: {accuracy} is below the minimum threshold of {MIN_TEST_ACCURACY}")
    print("Model passed quality gate.")

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    # Model registration
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="iris_model",
        registered_model_name="iris-classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    #Assign the champioin alias to the newly registered model
    client = MlflowClient()
    #Identify current training run
    #We use this run ID to find the model version registered by this particular run
    current_run_id = mlflow.active_run().info.run_id

    #Retrieve all versions registered under this model name
    registered_versions = client.search_model_versions("name='iris-classifier'")

    #Find the version created by the current training run
    current_model_version = next((version for version in registered_versions if version.run_id == current_run_id), None)

    client.set_registered_model_alias(
        name="iris-classifier",
        version=current_model_version.version,
        alias="champion"
    )

    print("Registered Model: iris_classifier")