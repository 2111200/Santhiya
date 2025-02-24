import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
df = pd.read_csv(r"D:\MLOps ass 01\Social_Network_Ads - Social_Network_Ads.csv")
print("Dataset Preview:")
print(df.head())
df.dropna(inplace=True)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 
X = df.drop(columns=['Purchased']) 
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['Purchased'] = y_train.values
train_df.to_csv(r"D:\MLOps ass 01\train.csv", index=False) 

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Purchased'] = y_test.values
test_df.to_csv(r"D:\MLOps ass 01\test.csv", index=False) 

print("Preprocessing complete. Train and test datasets saved successfully!")
