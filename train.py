import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(r"D:\MLOps ass 01\train.csv")  
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
pipeline_rf = Pipeline([("model", RandomForestClassifier(n_estimators=100, random_state=42))])
pipeline_lr = Pipeline([("model", LogisticRegression())])
pipeline_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
with open(r"D:\MLOps ass 01\random_forest.pkl", "wb") as f:
    pickle.dump(pipeline_rf, f)
with open(r"D:\MLOps ass 01\logistic_regression.pkl", "wb") as f:
    pickle.dump(pipeline_lr, f)
print("✅ Models trained and saved successfully!")
