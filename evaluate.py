import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, log_loss, confusion_matrix
test = pd.read_csv(r"D:\MLOps ass 01\test.csv")  # Adjusted for Social Network Ads dataset
X_test = test.iloc[:, :-1] 
y_test = test.iloc[:, -1]   
results = {}

for model_name, model_file in [
    ("RandomForest", r"D:\MLOps ass 01\logistic_regression.pkl"),
    ("LogReg", r"D:\MLOps ass 01\random_forest.pkl"),
]:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[model_name] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "log_loss": loss,
        "precision": precision,
        "confusion_matrix": cm.tolist(),
    }
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)
print("✅ Model evaluation complete. Results saved in 'metrics.json'.")
