import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

def preprocess_data(X, y):
    # balancear com SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    # normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    return X_scaled, y_bal, scaler
