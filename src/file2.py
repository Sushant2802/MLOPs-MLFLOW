import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


import dagshub
dagshub.init(repo_owner='Sushant2802', repo_name='MLOPs-MLFLOW', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Sushant2802/MLOPs-MLFLOW.mlflow")



# load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the params for RF model
max_depth = 8 
n_estimators = 5


# mention your experiment below
mlflow.set_experiment('Experiment_02')


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy ", accuracy)
    mlflow.log_param("Max_depth ", max_depth)
    mlflow.log_param("n_estimators ", n_estimators)

    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted ")
    plt.title("Confusion Matrix ")


    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)


    # tags
    mlflow.set_tags({"Author":"Sushant", "Project":"Wine Classification"})

    # log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")   # this gives lot of info like MLmodel, conda.yaml, model.pkl, python_env.yaml, requirement.txt, etc.

    print(accuracy)