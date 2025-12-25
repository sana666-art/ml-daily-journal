import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# --------------------------------------
# 1. Handle Missing Values
# --------------------------------------
def handle_missing_values(df, strategy="mean"):
    """
    strategy: mean, median, mode
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype != "object":
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# --------------------------------------
# 2. Encode Categorical Columns
# --------------------------------------
def encode_categorical(df):
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df


# --------------------------------------
# 3. Feature Scaling
# --------------------------------------
def scale_features(df, method="standard"):
    df = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


# --------------------------------------
# 4. Remove Outliers (IQR method)
# --------------------------------------
def remove_outliers(df):
    df = df.copy()

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
